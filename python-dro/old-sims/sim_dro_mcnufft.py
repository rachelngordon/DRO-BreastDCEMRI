import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import imageio # For creating GIFs
from einops import rearrange
import torch
import torch.nn as nn

# Import the custom functions from your other Python files
from dro import gen_dro
from mask import disp_mask
from kspace import gen_kspace_data
import nibabel as nib
import torchkbnufft as tkbn
from time import time
from scipy.ndimage import gaussian_filter

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
                    image = data[..., ii]
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
    NUM_SUBJECTS = 53
    SPOKES_PER_FRAME = 36
    NOISE_LEVEL = 0.05
    DEVICE = 'cpu'
    CREATE_DRO_GIF = True
    DISPLAY_MASKS = True

    # ===================================================================
    # --- 2. DATA COLLECTION (from .mat files) ---
    # ===================================================================
    print("--- Step 1: Loading Real Patient Data ---")
    patient_data_list = []
    print(f"Attempting to load data from: {os.path.abspath(DATA_DIRECTORY)}")

    for i in range(1, NUM_SUBJECTS + 1):
        filename = f'sub{i}.mat'
        full_path = os.path.join(DATA_DIRECTORY, filename)
        try:
            mat_contents = loadmat(full_path)
            patient_data_list.append({
                'ID': f'sub_{i}',
                'mask': mat_contents['mask'],
                'AIF': mat_contents['aif'].flatten(),
                'S0': mat_contents['S0'],
                'smap': mat_contents['smap']
            })
        except FileNotFoundError:
            print(f"Warning: Could not find file {filename}. Skipping.")
        except KeyError as e:
            print(f"Warning: File {filename} is missing a required key: {e}. Skipping.")

    if not patient_data_list:
        raise ValueError("No data was loaded. Please check DATA_DIRECTORY.")

    print(f"Successfully loaded {len(patient_data_list)} patient cases.\n")

    # ===================================================================
    # --- 3. GENERATE DIGITAL REFERENCE OBJECT (DRO) ---
    # ===================================================================
    print("--- Step 2: Generating DRO from a Randomly Selected Case ---")
    # dro_results = gen_dro(patient_data_list, option={}) 
    dro_results = gen_dro(patient_data_list, option={}) 
    simImg = dro_results['simImg']
    mask_dict = dro_results['mask']
    smap = dro_results['smap']
    S0 = dro_results['S0']
    print(f"DRO generation complete for case ID: {dro_results['ID']}")
    print(f"  - Simulated Image (simImg) shape: {simImg.shape}")
    print(f"  - Sensitivity Maps shape: {smap.shape}")


    # broadcast to coil images
    coil_images = simImg[:, :, np.newaxis, :] * smap[:, :, :, np.newaxis]

    # --- 3. Verify the Result ---
    print(f"Shape of simImg: {simImg.shape}")
    print(f"Shape of smap:   {smap.shape}")
    print("-" * 30)
    print(f"Shape of the final coil_images: {coil_images.shape}")
    print(f"Data type of the final coil_images: {coil_images.dtype}")


    # --- PLOTTING CODE ---

    # Choose a time frame to visualize (e.g., the 10th time frame)
    time_to_plot = 10
    C = 16

    # 1. Plot the Magnitude for all 16 coils
    fig_mag, axs_mag = plt.subplots(4, 4, figsize=(12, 12))
    fig_mag.suptitle(f'Magnitude of all 16 Coil Images at Time Frame {time_to_plot}', fontsize=16)

    for i in range(C):
        row = i // 4
        col = i % 4
        
        # Get the magnitude of the i-th coil at the chosen time
        magnitude_image = np.abs(coil_images[:, :, i, time_to_plot])
        
        ax = axs_mag[row, col]
        ax.imshow(magnitude_image, cmap='gray')
        ax.set_title(f'Coil {i+1}')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.show()


    # 2. Plot the Phase for all 16 coils
    fig_phase, axs_phase = plt.subplots(4, 4, figsize=(12, 12))
    fig_phase.suptitle(f'Phase of all 16 Coil Images at Time Frame {time_to_plot}', fontsize=16)

    for i in range(C):
        row = i // 4
        col = i % 4

        # Get the phase of the i-th coil at the chosen time
        phase_image = np.abs(coil_images[:, :, i, time_to_plot])
        
        ax = axs_phase[row, col]
        # 'hsv' or 'twilight' are good colormaps for cyclic data like phase
        ax.imshow(phase_image, cmap='gray')
        ax.set_title(f'Coil {i+1}')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('coil_images.png')


#    # --- Visualize the complex data (Magnitude and Phase) ---
#     # This new block will mask the phase plot for better clarity.

#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#     fig.suptitle("Real-Valued Simulated Image (Frame 4)", fontsize=16)

#     # Get a frame with some enhancement
#     frame_to_show = simImg[:, :, 4]
#     magnitude = np.abs(frame_to_show)
#     phase = np.angle(frame_to_show)

#     # --- Create a mask to hide phase in low-signal areas ---
#     # Set the threshold to a small fraction of the max magnitude
#     magnitude_mask = magnitude > (0.05 * magnitude.max())

#     # Apply the mask: where the mask is False, set phase to 0 (or np.nan)
#     phase_masked = np.zeros_like(phase)
#     phase_masked[magnitude_mask] = phase[magnitude_mask]


#     # Plot Magnitude
#     im1 = axes[0].imshow(magnitude, cmap='gray')
#     axes[0].set_title("Magnitude")
#     axes[0].axis('off')
#     fig.colorbar(im1, ax=axes[0])

#     # Plot the MASKED Phase
#     im2 = axes[1].imshow(phase_masked, cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)
#     axes[1].set_title("Phase (Masked)") # Updated title
#     axes[1].axis('off')
#     fig.colorbar(im2, ax=axes[1])

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig("complex_dro_mag_phase.png")
#     plt.close()



    # ===================================================================
    # --- 4. DISPLAY MASKS & GIF (if enabled) ---
    # ===================================================================
    if CREATE_DRO_GIF:
        print("\nCreating GIF of the dynamic DRO image series...")
        # ... GIF creation code ...
        print("Saved DRO animation to 'dro_simulation.gif'")
    if DISPLAY_MASKS:
        print("\n--- Step 3: Displaying Segmentation Masks ---")
        _, _ = disp_mask(mask_dict, S0, show_plot=True)
        print("Mask display generated.")



    # ===================================================================
    # --- 5. SIMULATE K-SPACE WITH TORCHKBNUFFT ---
    # ===================================================================

    # cuda:0
    # torch.Size([320, 320, 8])
    # torch.complex64
    # torch.Size([1, 16, 320, 320])
    # torch.complex64

    device = torch.device("cuda")
    N_samples = 640
    N_spokes = 36
    N_time = 21

    ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, N_spokes, N_time)
    ktraj = ktraj.to(device)
    dcomp = dcomp.to(device)
    nufft_ob = nufft_ob.to(device)
    adjnufft_ob = adjnufft_ob.to(device)

    physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)

    smap = rearrange(torch.tensor(smap), 'h w c -> c h w').unsqueeze(0).to(device)

    # set phase to zero in simulated image (will broadcast to csmaps in NUFFT)
    simImg = torch.tensor(simImg).to(torch.cfloat).to(device)


    sim_kspace = physics(False, simImg, smap)
        
    recon_img = physics(True, sim_kspace.to(device), smap)

    plt.imshow(np.abs(recon_img[..., 0]))
    plt.savefig("dro_recon_test.png")
    plt.close()
