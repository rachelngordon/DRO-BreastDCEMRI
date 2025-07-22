from kspace import gen_kspace_data
import numpy as np


# --- Example Usage ---
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print("--- Testing Python NUFFT Data Generation ---")
    
    # 1. Create dummy input data
    H, W = 64, 64
    T = 10
    C = 4
    spokes_per_frame = 13
    n_lvl = 0.05
    
    img_np = np.zeros((H, W, T), dtype=np.complex64)
    img_np[H//4:3*H//4, W//4:3*W//4, :] = 1.0
    
    y, x = np.ogrid[:H, :W]
    smap_np = np.zeros((H, W, C), dtype=np.complex64)
    smap_np[:,:,0] = (x + 1j*y) / (H + 1j*W)
    smap_np[:,:,1] = ((W-x) + 1j*y) / (H + 1j*W)

    # 2. Call the main function
    kspace, traj, test_img = gen_kspace_data(img_np, smap_np, spokes_per_frame, n_lvl)
    
    # 3. Print shapes to verify
    print(f"\nInput image shape: {img_np.shape}")
    print(f"Final k-space shape: {kspace.shape}")
    print(f"Trajectory shape: {traj.shape}")
    print(f"Test (adjoint) image shape: {test_img.shape}")
    
    # Expected shape: (nx, spokes*frames, ncoil)
    expected_k_shape = (H*2, spokes_per_frame*T, C)
    print(f"Expected k-space shape: {expected_k_shape}")
    assert kspace.shape == expected_k_shape, "Final k-space shape is incorrect!"
    
    # 4. Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(np.real(traj.flatten()), np.imag(traj.flatten()), s=0.1)
    axes[0].set_title('K-space Trajectory')
    axes[0].set_aspect('equal')
    
    axes[1].imshow(np.log(np.abs(kspace[:, :, 0]) + 1e-9), cmap='gray')
    axes[1].set_title('Log of K-space (Coil 0)')
    
    axes[2].imshow(np.abs(test_img[:, :, -1]), cmap='gray')
    axes[2].set_title('Adjoint Recon (Last Frame)')
    
    plt.tight_layout()
    plt.savefig("kspace_out.png")