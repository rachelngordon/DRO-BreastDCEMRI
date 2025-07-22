import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import imageio # For creating GIFs
from einops import rearrange
import torch
import torch.nn as nn
import nibabel as nib
import torchkbnufft as tkbn
from time import time
from scipy.ndimage import gaussian_filter

# Import the custom functions from your other Python files
# Note: These functions are included directly in this script as per the original code.
from dro import gen_dro
from mask import disp_mask
from kspace import gen_kspace_data


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

                    x_temp = self.adjnufft_ob(kd*d, k, smaps=smaps)

                    x[:, :, ii] = torch.squeeze(x_temp) / np.sqrt(Nx * Ny)

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



# --- Main execution block for the full simulation pipeline ---
if __name__ == '__main__':

    # ===================================================================
    # --- 1. CONFIGURATION ---
    # ===================================================================
    DATA_DIRECTORY = '/gpfs/data/karczmar-lab/workspaces/rachelgordon/DRO-BreastDCEMRI/data'
    OUTPUT_DIRECTORY = '/ess/scratch/scratch1/rachelgordon/simulated_dataset/'
    NUM_SUBJECTS = 53
    # NUFFT parameters (N_TIME_FRAMES is now removed)
    N_SAMPLES_SPOKE = 640
    N_SPOKES_FRAME = 36
    
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print("CUDA is available. Using GPU.")
    else:
        DEVICE = 'cpu'
        print("CUDA not available. Using CPU.")
    
    device = torch.device(DEVICE)
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # ===================================================================
    # --- 2. DATA COLLECTION (from .mat files) ---
    # ===================================================================
    # (This part is unchanged)
    print(f"--- Step 1: Loading Real Patient Data ({NUM_SUBJECTS} subjects) ---")
    patient_data_list = []
    for i in range(1, NUM_SUBJECTS + 1):
        filename = f'sub{i}.mat'
        full_path = os.path.join(DATA_DIRECTORY, filename)
        try:
            mat_contents = loadmat(full_path)
            patient_data_list.append({
                'ID': f'sub{i}', 'mask': mat_contents['mask'], 'AIF': mat_contents['aif'].flatten(),
                'S0': mat_contents['S0'], 'smap': mat_contents['smap']
            })
        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: Could not process {filename}: {e}. Skipping.")
    if not patient_data_list: raise ValueError("No data was loaded.")
    print(f"Successfully loaded {len(patient_data_list)} patient cases.\n")

    # ===================================================================
    # --- 3. SIMULATION LOOP FOR ALL SAMPLES (RESTRUCTURED) ---
    # ===================================================================
    print(f"--- Step 2: Starting Dynamic Simulation Loop for {len(patient_data_list)} Samples ---")
    
    for i, current_patient_data in enumerate(patient_data_list):
        case_num = i + 1
        patient_id = current_patient_data['ID']
        print(f"\n--- Processing Sample {case_num}/{len(patient_data_list)} (Source: {patient_id}) ---")

        try:
            # --- A. GENERATE DRO (THIS DETERMINES NUM_FRAMES) ---
            dro_results = gen_dro([current_patient_data], option={})
            simImg_np = dro_results['simImg']
            smap_np = dro_results['smap']

            parMap_gt = dro_results['parMap']
            aif_gt = dro_results['aif']
            S0_gt = dro_results['S0']
            T10_gt = dro_results['T10']
            mask = dro_results['mask']
            
            # Get num_frames dynamically from the generated image
            num_frames = simImg_np.shape[2]
            print(f"  Dynamically detected {num_frames} frames from AIF.")

            # --- B. PREPARE NUFFT (NOW INSIDE THE LOOP) ---
            ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_SAMPLES_SPOKE, N_SPOKES_FRAME, num_frames)
            
            # Instantiate the physics model for this specific case
            physics = MCNUFFT(nufft_ob.to(device), adjnufft_ob.to(device), ktraj.to(device), dcomp.to(device))
            
            # --- C. SIMULATE K-SPACE & RECONSTRUCT ---
            smap_torch = rearrange(torch.tensor(smap_np), 'h w c -> c h w').unsqueeze(0).to(device)
            simImg_torch = torch.tensor(simImg_np).to(torch.cfloat).to(device)
            
            sim_kspace_torch = physics(False, simImg_torch, smap_torch)
            recon_img_torch = physics(True, sim_kspace_torch.to(device), smap_torch)

            # --- D. SAVE ALL DATA TO DISK ---
            sample_dir = os.path.join(OUTPUT_DIRECTORY, f"sample_{case_num:03d}_{patient_id}")
            os.makedirs(sample_dir, exist_ok=True)

            
            dro_to_save = simImg_torch.cpu().numpy()
            kspace_to_save = sim_kspace_torch.cpu().numpy()
            recon_to_save = recon_img_torch.cpu().numpy()


            print(f"  Saving data to: {sample_dir}")
            print("kspace shape: ", kspace_to_save.shape)
            print("kspace dtype: ", kspace_to_save.dtype)


            # ==========================================================
            # --- STEP 1: CONVERT THE MASK DICTIONARY TO AN INTEGER ARRAY ---
            # ==========================================================

            # print("-> Converting mask dictionary to a single integer array for saving...")

            # # Define the integer ID for each tissue type. 0 is background.
            # TISSUE_ID_MAPPING = {
            #     'glandular': 1,
            #     'benign': 2,
            #     'malignant': 3,
            #     'muscle': 4,
            #     'skin': 5,
            #     'liver': 6,
            #     'heart': 7,
            #     'vascular': 8
            # }

            # # Initialize a new array with zeros (background)
            # height, width = S0_gt.shape
            # integer_mask_array = np.zeros((height, width), dtype=np.int32)

            # # Loop through the mapping and fill the integer array.
            # # The order matters: if a pixel is in multiple masks (unlikely but possible),
            # # the last one processed will overwrite the previous ones.
            # for tissue_name, tissue_id in TISSUE_ID_MAPPING.items():
            #     if tissue_name in mask and mask[tissue_name].any():
            #         # Get the boolean mask for the current tissue
            #         boolean_mask = mask[tissue_name]
            #         # Where this mask is True, set the integer ID
            #         integer_mask_array[boolean_mask] = tissue_id
                    
            # print("   Conversion complete. Ready to save.")

            output_filepath_npz = os.path.join(sample_dir, 'dro_ground_truth.npz')

            np.savez_compressed(
                output_filepath_npz,

                # --- Ground Truth Data from DRO ---
                ground_truth_images=dro_to_save,  # The perfect, fully-sampled images
                parMap=parMap_gt,           # The true PK parameters (H, W, 4)
                aif=aif_gt,                 # The Arterial Input Function (Time,)
                S0=S0_gt,                   # The S0 map (H, W)
                T10=T10_gt,                 # The T10 map (H, W)
                # mask=integer_mask_array

                **mask
            )

            print("dro saved to: ", output_filepath_npz)
            
            # np.save(os.path.join(sample_dir, 'dro_ground_truth.npy'), dro_to_save)
            np.save(os.path.join(sample_dir, 'simulated_kspace.npy'), kspace_to_save)
            np.save(os.path.join(sample_dir, 'reconstructed_image.npy'), recon_to_save)
            np.save(os.path.join(sample_dir, 'csmaps.npy'), smap_np)
            
            print(f"  Successfully saved all files for sample {case_num}.")

        except Exception as e:
            print(f"\n!!!!!! ERROR processing Sample {case_num} ({patient_id}) !!!!!!!")
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n=========================================================")
    print("          BATCH SIMULATION COMPLETE")
    print(f"  Total samples processed: {len(patient_data_list)}")
    print(f"  Data saved in: {os.path.abspath(OUTPUT_DIRECTORY)}")
    print("=========================================================")