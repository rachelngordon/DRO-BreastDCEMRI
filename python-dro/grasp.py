import numpy as np
from einops import rearrange
import torch
import sigpy as sp
import matplotlib.pyplot as plt
import os

from sigpy.mri import app


def get_traj(N_spokes=13, N_time=1, base_res=320, gind=1):

    N_tot_spokes = N_spokes * N_time

    N_samples = base_res * 2

    base_lin = np.arange(N_samples).reshape(1, -1) - base_res

    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (gind + tau - 1)

    base_rot = np.arange(N_tot_spokes).reshape(-1, 1) * base_rad

    traj = np.zeros((N_tot_spokes, N_samples, 2))
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin

    traj = traj / 2

    traj = traj.reshape(N_time, N_spokes, N_samples, 2)

    return np.squeeze(traj)



def grasp_recon(sample_id, spokes_per_frame, data_path):

    sample_path = f'{data_path}sample_{sample_id:03d}_sub{sample_id}/'

    # load k-space
    kspace = np.load(f'{sample_path}simulated_kspace.npy')
    kspace = rearrange(torch.tensor(kspace), 'c (sp sam) t -> t c sp sam', sp=36)
    kspace = np.expand_dims(kspace, axis=[1,3])

    # load csmaps
    csmaps = np.load(f'{sample_path}csmaps.npy')
    csmaps = rearrange(csmaps, 'h w c -> c h w')
    csmaps = np.expand_dims(csmaps, axis=1)



    traj = get_traj(N_spokes=spokes_per_frame, N_time=kspace.shape[0])

    device = sp.Device(0 if torch.cuda.is_available() else -1)


    # reconstruct image
    R1 = app.HighDimensionalRecon(kspace, csmaps,
                            combine_echo=False,
                            lamda=0.001,
                            coord=traj,
                            regu='TV', regu_axes=[0],
                            max_iter=10,
                            solver='ADMM', rho=0.1,
                            device=device,
                            show_pbar=False,
                            verbose=False).run()

    R1 = np.squeeze(R1.get())

    out_dir = os.path.join(sample_path, 'grasp_recon.npy')

    np.save(out_dir, R1)

    print(f"recon image for sample {sample_id} saved to: ", out_dir)
    print("recon image shape: ", R1.shape)

    return R1


spokes_per_frame = 36
data_path = '/ess/scratch/scratch1/rachelgordon/simulated_dataset/'

for i in range(1, 54):
    grasp_recon(i, spokes_per_frame, data_path)


# # ===================================================================
# # --- 4. VISUALIZE THE DRO IMAGES WITH HIGH QUALITY ---
# # ===================================================================

# print("\nGenerating high-quality plot for DRO images...")

# # --- Configuration for the Plot Layout ---
# # You can change this to control how many images appear per row.
# # For 21 frames, 7 columns is a good choice.
# COLS_PER_ROW = 7
# NUM_TIME_FRAMES = R1.shape[0]
# # Calculate the number of rows needed
# num_rows = (NUM_TIME_FRAMES + COLS_PER_ROW - 1) // COLS_PER_ROW # This is a ceiling division

# # --- Create the figure and axes ---
# # We make the figure taller to accommodate multiple rows.
# fig, axes = plt.subplots(num_rows, COLS_PER_ROW, figsize=(COLS_PER_ROW * 3, num_rows * 3.5))
# fig.suptitle(f"Undersampled GRASP Reconstructions for Sample {sample_id}", fontsize=16)

# # --- IMPORTANT: Calculate a consistent brightness/contrast scale ---
# # We find the min and max signal intensity across ALL time frames.
# # Using a consistent scale is essential to visually see the enhancement.
# vmin = np.min(np.abs(R1))
# vmax = np.max(np.abs(R1))
# print(f"Using consistent scaling for all images: vmin={vmin:.2f}, vmax={vmax:.2f}")


# # Flatten the axes array to make it easy to iterate through
# # This handles any number of rows/columns automatically.
# axes_flat = axes.flatten()

# for t in range(NUM_TIME_FRAMES):
#     ax = axes_flat[t]
    
#     # Display the ground truth DRO for the current time frame
#     # We use the calculated vmin and vmax for consistent scaling.
#     img = ax.imshow(np.abs(R1[t]), cmap='gray', vmin=vmin, vmax=vmax)
    
#     # Add a title to each subplot indicating the time frame
#     ax.set_title(f"Time Frame = {t}")
    
#     # Turn off the pixel coordinate axes for a cleaner look
#     ax.axis('off')

# # --- Clean up any unused subplots ---
# # If you have 21 frames and a 3x7 grid, there are no unused plots.
# # If you had 20 frames, this would turn off the 21st plot.
# for i in range(NUM_TIME_FRAMES, len(axes_flat)):
#     axes_flat[i].axis('off')

# # --- Adjust layout and show the plot ---
# # `tight_layout` cleans up spacing between plots.
# # The `rect` argument prevents the suptitle from overlapping the subplots.
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.savefig("grasp_recon_sim_kspace.png")