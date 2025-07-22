import numpy as np
from nufft import MCNUFFT
import torch

def trajectory_golden_angle_grog(num_total_views, num_points_per_spoke):
    """
    Generates a 2D golden-angle radial trajectory.
    
    This is a Python translation of the Trajectory_GoldenAngle_GROG MATLAB function.

    Args:
        num_total_views (int): The total number of spokes (views) to generate.
        num_points_per_spoke (int): The number of points along each spoke.

    Returns:
        np.ndarray: A complex-valued array of shape (num_points_per_spoke, num_total_views)
                    representing the k-space trajectory coordinates.
    """
    # Golden ratio conjugate
    golden_ratio_conjugate = (np.sqrt(5) + 1) / 2
    golden_angle_rads = np.pi / golden_ratio_conjugate
    
    # Generate angles for each spoke
    angles = np.arange(num_total_views) * golden_angle_rads
    
    # Generate the radial sampling points (Rho) from -N/2 to N/2
    # The MATLAB version's `+0.5` creates centered samples.
    # np.linspace is a clearer way to achieve this.
    rho = np.linspace(
        -(num_points_per_spoke / 2) + 0.5,
        (num_points_per_spoke / 2) - 0.5,
        num_points_per_spoke
    )
    
    # Use broadcasting to create the full trajectory without a loop
    # rho shape: (num_points, 1)
    # angles shape: (1, num_views)
    k_x = rho[:, np.newaxis] * np.sin(angles[np.newaxis, :])
    # The MATLAB version has a negative sign for Y. We replicate that here.
    k_y = -rho[:, np.newaxis] * np.cos(angles[np.newaxis, :])
    
    # Combine into a complex trajectory
    trajectory = k_x + 1j * k_y
    
    return trajectory


def gen_kspace_data(img, smap, spokes_per_frame, n_lvl, device='cpu'):
    """
    Simulates the acquisition of MRI k-space data from a dynamic image series.

    This is a Python translation of the gen_kspace_data MATLAB function.
    
    Args:
        img (np.ndarray): Input image series. Shape: (H, W, T).
        smap (np.ndarray): Coil sensitivity maps. Shape: (H, W, C).
        spokes_per_frame (int): Number of radial spokes per time frame.
        n_lvl (float): Level of Gaussian noise to add.
        device (str): Device for torch computation ('cpu' or 'cuda:X').
        
    Returns:
        tuple: A tuple containing:
            kspace_final (np.ndarray): Simulated k-space data. Shape: (nx, spokes*frames, ncoil).
            traj_final (np.ndarray): The full k-space trajectory used.
            test_img (np.ndarray): The image from the adjoint NUFFT operation.
    """
    # Numerical stability
    img[img == 0] = 1e-8
    
    # --- 1. Setup trajectory and density compensation (DCF) ---
    img_h, _, num_frames = img.shape
    num_coils = smap.shape[2]
    
    # CRITICAL: Trajectory is generated on a 2x oversampled grid
    nx = img_h * 2
    num_total_views = spokes_per_frame * num_frames
    
    # Generate the trajectory and normalize it to the range [-0.5, 0.5]
    traj = trajectory_golden_angle_grog(num_total_views, nx) / nx
    
    # Create a simple ramp DCF for each spoke
    dcf_1d = np.abs(np.linspace(-1, 1, nx))
    dcf = np.tile(dcf_1d[:, np.newaxis], (1, num_total_views))
    
    # Reshape trajectory and DCF into (points, spokes_per_frame, num_frames)
    traju = traj.reshape((nx, spokes_per_frame, num_frames), order='F')
    dcfu = dcf.reshape((nx, spokes_per_frame, num_frames), order='F')
    
    # --- 2. Initialize NUFFT Operator and Perform Forward Operation ---
    # Normalize sensitivity maps
    smap_normalized = smap / np.max(np.abs(smap))
    
    # Instantiate our Python MCNUFFT class
    nufft_operator = MCNUFFT(k=traju, w=dcfu, b1=smap_normalized, device=device)
    
    # Perform the forward NUFFT (image -> k-space)
    # Output shape: (T, C, M) where M = nx * spokes_per_frame
    kspace_torch = nufft_operator.forward(img)
    
    # --- 3. Add Noise and Apply a final (unusual) normalization ---
    # Generate complex Gaussian noise
    noise_real = torch.randn_like(kspace_torch.real)
    noise_imag = torch.randn_like(kspace_torch.imag)
    kspace_torch = kspace_torch + n_lvl * (noise_real + 1j * noise_imag)
    
    # NOTE: The following line from the MATLAB code is unusual. It divides the k-space
    # data by the sqrt of the DCF. This is not standard practice (DCF is usually
    # applied in the adjoint). We replicate it here for fidelity.
    # We get the sqrt(dcf) from the operator's stored 'w' and reshape it.
    # sqrt_dcf_torch = nufft_operator.w.reshape(nx, spokes_per_frame, num_frames, order='F')
    sqrt_dcf_torch = nufft_operator.w
    sqrt_dcf_torch = sqrt_dcf_torch.permute(2, 0, 1).flatten(start_dim=1) # (T, M)
    # Add a coil dimension for broadcasting
    kspace_torch = kspace_torch / sqrt_dcf_torch.unsqueeze(1) # (T, 1, M)

    # --- 4. Perform Adjoint Operation and Final Reshaping ---
    # Perform the adjoint NUFFT (k-space -> image) for testing
    test_img_torch = nufft_operator.adjoint(kspace_torch)
    test_img = test_img_torch.cpu().detach().numpy()
    
    # Replicate the final permute and reshape from the MATLAB script
    # Current k-space shape: (T, C, M) where M = nx * spokes_per_frame
    # First, un-flatten the last dimension -> (T, C, nx, spokes_per_frame)
    kspace_reshaped = kspace_torch.reshape(num_frames, num_coils, nx, spokes_per_frame)
    # Permute to (nx, spokes_per_frame, num_frames, num_coils)
    kspace_permuted = kspace_reshaped.permute(2, 3, 0, 1)
    # Finally, reshape to (nx, spokes_per_frame * num_frames, num_coils)
    kspace_final = kspace_permuted.reshape(nx, spokes_per_frame * num_frames, num_coils)
    
    return kspace_final.cpu().detach().numpy(), traj, test_img
