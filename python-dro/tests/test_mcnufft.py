
import matplotlib.pyplot as plt
import torch
from einops import rearrange
import numpy as np
import torchkbnufft as tkbn
from time import time
import glob
import os
import torch
import torch.nn as nn
import numpy as np

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import json


class SliceDataset(Dataset):
    """
    A Dataset that:
      - Looks for all .h5/.hdf5 files under `root_dir`.
      - Each file is assumed to contain a dataset at `dataset_key`, with shape (... Z),
        where Z is the number of slices/partitions.
      - Splits each volume into Z separate examples (one per slice).
      - Returns each slice as a torch.Tensor.
    """

    def __init__(
        self,
        root_dir,
        patient_ids,
        dataset_key="kspace",
        file_pattern="*.h5",
        slice_idx=41,
        N_time = 8,
        N_coils=16
    ):
        """
        Args:
            root_dir (str): Path to the folder containing all HDF5 k-space files.
            dataset_key (str): The key/path inside each .h5 file to the k-space dataset (e.g. "kspace").
            file_pattern (str): Glob pattern to match your HDF5 files (default "*.h5").
        """
        super().__init__()
        self.root_dir = root_dir
        self.dataset_key = dataset_key
        self.slice_idx = slice_idx
        self.N_time = N_time
        self.N_coils = N_coils

        # Find all matching HDF5 files under root_dir
        all_files = sorted(glob.glob(os.path.join(root_dir, file_pattern)))
        if len(all_files) == 0:
            raise RuntimeError(
                f"No files found in {root_dir} matching pattern {file_pattern}"
            )

        # filter file list by patient ID substring
        filtered = []
        for fp in all_files:
            fname = os.path.basename(fp)
            # Check if any patient_id appears in the filename
            if any(pid in fname for pid in patient_ids):
                filtered.append(fp)
        self.file_list = filtered

        if len(self.file_list) == 0:
            raise RuntimeError("No files matched the provided patient_ids filter.")

        # Build a list of (file_path, slice_index) for every slice in every volume
        # self.slice_index_map = []
        # for fp in self.file_list:
        #     with h5py.File(fp, "r") as f:
        #         if self.dataset_key not in f:
        #             raise KeyError(f"Dataset key '{self.dataset_key}' not found in file {fp}")
        #         ds = f[self.dataset_key]
        #         num_slices = ds.shape[0]

        #     for z in range(num_slices):
        #         self.slice_index_map.append((fp, z))

    def load_dynamic_img(self, patient_id):

        H = W = 320
        data = np.empty((2, self.N_time, H, W), dtype=np.float32)

        for t in range(self.N_time):
            # load image 
            img_path = f'/ess/scratch/scratch1/rachelgordon/dce-{self.N_time}tf/{patient_id}/slice_{self.slice_idx:03d}_frame_{t:03d}.nii'

            # if os.path.exists(img_path):

            img = nib.load(img_path)
            img_data = img.get_fdata()

            if img_data.shape != (2, H, W):
                raise ValueError(f"{img_path} has shape {img_data.shape}; "
                                f"expected (2, {H}, {W})")

            data[:, t] = img_data.astype(np.float32)
            
            # else:
            #     return None

        return torch.from_numpy(data) 
    
    def load_csmaps(self, patient_id):

        ground_truth_dir = os.path.join(os.path.dirname(self.root_dir), 'cs_maps')
        csmap_path = os.path.join(ground_truth_dir, patient_id + '_cs_maps', f'cs_map_slice_{self.slice_idx:03d}.npy')

        csmap = np.load(csmap_path)

        return csmap.squeeze()


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Returns a single slice of k-space as a torch.Tensor.
        The output shape will be the standard (C=2, T, S, I) where C is [real, imag].
        """

        # load GRASP recon image
        file_path = self.file_list[idx]
        patient_id = file_path.split('/')[-1].strip('.h5')

        grasp_img = self.load_dynamic_img(patient_id)
        csmap = self.load_csmaps(patient_id)

        with h5py.File(file_path, "r") as f:
            ds = torch.tensor(f[self.dataset_key][:])
            kspace_slice = ds[self.slice_idx]

        # Select the first coil
        if self.N_coils == 1:
            kspace_slice = kspace_slice[:, 0, :, :]  # Shape: (T, S, I)

        # Separate real and imaginary components
        real_part = kspace_slice.real
        imag_part = kspace_slice.imag

        # Stack them along a new 'channel' dimension (dim=0).
        # This creates the final, standard (C=2, T, S, I) format.
        kspace_final = torch.stack([real_part, imag_part], dim=0).float()

        # The final shape is (2, num_timeframes, num_spokes, num_samples)
        # e.g., (2, 8, 36, 640)
        return kspace_final, csmap, grasp_img


dtype = torch.complex64

def trajGR(Nkx, Nspokes):
    '''
    function for generating golden-angle radial sampling trajectory
    :param Nkx: spoke length
    :param Nspokes: number of spokes
    :return: ktraj: golden-angle radial sampling trajectory
    '''
    # ga = np.deg2rad(180 / ((np.sqrt(5) + 1) / 2))
    ga = np.pi * ((1 - np.sqrt(5)) / 2)
    kx = np.zeros(shape=(Nkx, Nspokes))
    ky = np.zeros(shape=(Nkx, Nspokes))
    ky[:, 0] = np.linspace(-np.pi, np.pi, Nkx)
    for i in range(1, Nspokes):
        kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
        ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]
    ky = np.transpose(ky)
    kx = np.transpose(kx)

    ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)

    return ktraj

################### prepare NUFFT ################
def prep_nufft(Nsample, Nspokes, Ng):

    overSmaple = 2
    im_size = (int(Nsample/overSmaple), int(Nsample/overSmaple))
    grid_size = (Nsample, Nsample)

    ktraj = trajGR(Nsample, Nspokes * Ng)
    ktraj = torch.tensor(ktraj, dtype=torch.float)
    dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj, im_size=im_size)
    dcomp = dcomp.squeeze()

    ktraju = np.zeros([2, Nspokes * Nsample, Ng], dtype=float)
    dcompu = np.zeros([Nspokes * Nsample, Ng], dtype=complex)

    for ii in range(0, Ng):
        ktraju[:, :, ii] = ktraj[:, (ii * Nspokes * Nsample):((ii + 1) * Nspokes * Nsample)]
        dcompu[:, ii] = dcomp[(ii * Nspokes * Nsample):((ii + 1) * Nspokes * Nsample)]

    ktraju = torch.tensor(ktraju, dtype=torch.float)
    dcompu = torch.tensor(dcompu, dtype=torch.complex64)

    nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size)  # forward nufft
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size)  # backward nufft

    return ktraju, dcompu, nufft_ob, adjnufft_ob


class MCNUFFT(nn.Module):
    def __init__(self, nufft_ob, adjnufft_ob, ktraj, dcomp):
        super(MCNUFFT, self).__init__()
        self.nufft_ob = nufft_ob
        self.adjnufft_ob = adjnufft_ob
        self.ktraj = torch.squeeze(ktraj)
        self.dcomp = torch.squeeze(dcomp)

    def forward(self, inv, data, smaps):

        data = torch.squeeze(data)  # delete redundant dimension
        Nx = smaps.shape[2]
        Ny = smaps.shape[3]

        if inv:  # adjoint nufft

            smaps = smaps.to(dtype)

            if len(data.shape) > 2:  # multi-frame

                x = torch.zeros([Nx, Ny, data.shape[2]], dtype=dtype)

                for ii in range(0, data.shape[2]):
                    kd = data[:, :, ii]
                    k = self.ktraj[:, :, ii]
                    d = self.dcomp[:, ii]

                    kd = kd.unsqueeze(0)
                    d = d.unsqueeze(0).unsqueeze(0)

                    tt1 = time()

                    x_temp = self.adjnufft_ob(kd*d, k, smaps=smaps)

                    x[:, :, ii] = torch.squeeze(x_temp) / np.sqrt(Nx * Ny)
                    tt2 = time()
                    # print('adjnufft time is %.6f' % (tt2 - tt1))

                    # plt.figure()
                    # plt.imshow(np.abs(x_temp.numpy()), 'gray')
                    # plt.show()

            else:  # single frame

                kd = data.unsqueeze(0)
                d = self.dcomp.unsqueeze(0).unsqueeze(0)
                x = self.adjnufft_ob(kd*d, self.ktraj, smaps=smaps)
                x = torch.squeeze(x) / np.sqrt(Nx * Ny)

        else:  # forward nufft

            if len(data.shape) > 2:  # multi-frame

                x = torch.zeros([smaps.shape[1], self.ktraj.shape[1], data.shape[-1]], dtype=dtype)

                for ii in range(0, data.shape[-1]):
                    image = data[:, :, ii]
                    k = self.ktraj[:, :, ii]

                    image = image.unsqueeze(0).unsqueeze(0)
                    x_temp = self.nufft_ob(image, k, smaps=smaps)
                    x[:, :, ii] = torch.squeeze(x_temp) / np.sqrt(Nx * Ny)

            else:  # single frame

                image = data.unsqueeze(0).unsqueeze(0)
                x = self.nufft_ob(image, self.ktraj, smaps=smaps)
                x = torch.squeeze(x) / np.sqrt(Nx * Ny)

        return x
    


# ----------------------------
# Example usage:
# ----------------------------
if __name__ == "__main__":
    # 1) Point this to wherever your HDF5 k-space files live
    root_dir = "/ess/scratch/scratch1/rachelgordon/dce-8tf/binned_kspace"

    # 2) If your HDF5 file stores k-space under a different key path, adjust dataset_key:
    dataset_key = "ktspace"  # change if your HDF5 group/dataset is named differently

    # 3) (Optional) Example transform: convert two‐channel real/imag → complex64
    def to_complex(x_np: "np.ndarray") -> "np.ndarray":
        """
        If x_np.shape = (C, H, W, 2) or similar where the last dim is [real, imag],
        convert to complex64 with shape (C, H, W).
        Adjust slicing logic if your real/imag channels are elsewhere.
        """
        real = x_np[..., 0].astype("float32")
        imag = x_np[..., 1].astype("float32")
        return (real + 1j * imag).astype("complex64")
    
    split_file = "/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/patient_splits.json"

    with open(split_file, "r") as fp:
        splits = json.load(fp)

    dataset = SliceDataset(
        root_dir=root_dir, dataset_key=dataset_key, file_pattern="*.h5", patient_ids=splits["train"][:5]
    )

    # 4) Wrap in DataLoader
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 5) Iterate and inspect
    for measured_kspace, csmap, grasp_img in loader:
        # kspace_batch.dtype could be torch.float32 or torch.complex64, depending on transform
        print(
            f"Batch: k-space batch shape = {measured_kspace.shape}, dtype = {measured_kspace.dtype}"
        )

        grasp_img = grasp_img.squeeze()

        grasp_img_complex = grasp_img[0] + 1j*grasp_img[1]

        grasp_img_complex = rearrange(grasp_img_complex, 't h w -> h w t')

        grasp_img_complex = torch.flip(grasp_img_complex, dims=[0])


        device = torch.device("cuda")
        N_samples = 640
        N_spokes = 36
        N_time = 8

        ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, N_spokes, N_time)
        ktraj = ktraj.to(device)
        dcomp = dcomp.to(device)
        nufft_ob = nufft_ob.to(device)
        adjnufft_ob = adjnufft_ob.to(device)

        physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)
        csmap = csmap.to(device).to(grasp_img_complex.dtype)

        print(csmap.device)

        print(grasp_img_complex.shape)
        print(grasp_img_complex.dtype)

        print(csmap.shape)
        print(csmap.dtype)

        sim_kspace = physics(False, grasp_img_complex.to(device), csmap)
        
        recon_img = physics(True, sim_kspace.to(device), csmap)

        print(recon_img.shape)

        plt.imshow(np.abs(recon_img[..., 0]))
        plt.savefig("recon_mcnufft_test.png")

        break
