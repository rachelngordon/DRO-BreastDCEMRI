import numpy as np
# Make sure your MCNUFFT class is importable
from nufft import MCNUFFT 
import torch

# The trajectory_golden_angle_grog function is correct as you wrote it.
def trajectory_golden_angle_grog(num_total_views, num_points_per_spoke):
    # (Your implementation is correct, so it is omitted here for brevity)
    golden_ratio_conjugate = (np.sqrt(5) + 1) / 2
    golden_angle_rads = np.pi / golden_ratio_conjugate
    angles = np.arange(num_total_views) * golden_angle_rads
    rho = np.linspace(
        -(num_points_per_spoke / 2) + 0.5,
        (num_points_per_spoke / 2) - 0.5,
        num_points_per_spoke
    )
    k_x = rho[:, np.newaxis] * np.sin(angles[np.newaxis, :])
    k_y = -rho[:, np.newaxis] * np.cos(angles[np.newaxis, :])
    trajectory = k_x + 1j * k_y
    return trajectory

# In your kspace.py file

def gen_kspace_data(img, smap, spokes_per_frame, n_lvl, device='cpu'):
    """
    Simulates the acquisition of MRI k-space data from a dynamic image series.
    (FINAL, PHYSICALLY-CONSISTENT VERSION)
    """
    img[img == 0] = 1e-8
    
    # --- 1. Setup trajectory and density compensation (DCF) ---
    img_h, _, num_frames = img.shape
    num_coils = smap.shape[2]
    nx = img_h * 2
    num_total_views = spokes_per_frame * num_frames
    
    # Generate and normalize trajectory to [-0.5, 0.5]
    traj = trajectory_golden_angle_grog(num_total_views, nx) / nx
    
    dcf_1d = np.abs(np.linspace(-1, 1, nx))
    dcf = np.tile(dcf_1d[:, np.newaxis], (1, num_total_views))
    
    traju = traj.reshape((nx, spokes_per_frame, num_frames), order='F')
    dcfu = dcf.reshape((nx, spokes_per_frame, num_frames), order='F')
    
    # --- 2. Initialize NUFFT Operator and Perform Forward Operation ---
    smap_normalized = smap / np.max(np.abs(smap))
    nufft_operator = MCNUFFT(k=traju, w=dcfu, b1=smap_normalized, device=device)
    kspace_torch = nufft_operator.forward(img)
    
    # --- 3. Add Noise ---
    # Generate complex Gaussian noise. The 1/sqrt(2) correctly scales it.
    noise = torch.randn_like(kspace_torch) * (n_lvl / np.sqrt(2))
    kspace_torch = kspace_torch + noise
    
    # ========================= CRITICAL CHANGE =========================
    # The non-standard k-space whitening has been REMOVED entirely.
    # The k-space data is now physically consistent.
    # ===================================================================

    # --- 4. Perform Adjoint Operation and Final Reshaping ---
    # The adjoint is now calculated on the clean, physical k-space data.
    test_img_torch = nufft_operator.adjoint(kspace_torch)
    test_img = test_img_torch.cpu().detach().numpy()
    
    # Reshape final k-space to be consistent with the rest of your pipeline
    kspace_final_permuted = kspace_torch.reshape(num_frames, num_coils, nx, spokes_per_frame).permute(2, 3, 0, 1)
    kspace_final = kspace_final_permuted.reshape(nx, num_total_views, num_coils)
    
    return kspace_final.cpu().detach().numpy(), traj, test_img