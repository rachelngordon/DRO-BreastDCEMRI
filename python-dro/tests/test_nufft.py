from nufft import MCNUFFT
import numpy as np
import matplotlib.pyplot as plt

# --- Example Usage and Testing (This part does not need to change) ---
if __name__ == '__main__':
    print("--- Testing MCNUFFT Python Class (Corrected API) ---")
    
    # 1. Create dummy input data
    H, W = 64, 64   # Image size
    C = 4           # Number of coils
    T = 10          # Number of time frames
    SPOKES = 13     # Spokes per frame
    POINTS = H      # Points per spoke
    
    # Dummy coil sensitivity maps (b1)
    y, x = np.ogrid[:H, :W]
    smap_np = np.zeros((H, W, C), dtype=np.complex64)
    smap_np[:,:,0] = (x + 1j*y) / (H + 1j*W)
    smap_np[:,:,1] = ((W-x) + 1j*y) / (H + 1j*W)
    smap_np[:,:,2] = (x + 1j*(H-y)) / (H + 1j*W)
    smap_np[:,:,3] = ((W-x) + 1j*(H-y)) / (H + 1j*W)
    
    # Dummy trajectory (k) and density compensation weights (w)
    golden_angle = np.pi * (3 - np.sqrt(5))
    k_all_frames = np.zeros((POINTS, SPOKES, T), dtype=np.complex64)
    radius = np.linspace(-0.5, 0.5, POINTS, endpoint=False)
    for t in range(T):
        angles = (np.arange(SPOKES) + t * SPOKES) * golden_angle
        k_real = radius[:, np.newaxis] * np.cos(angles[np.newaxis, :])
        k_imag = radius[:, np.newaxis] * np.sin(angles[np.newaxis, :])
        k_all_frames[:, :, t] = k_real + 1j * k_imag
    
    # Simple density compensation (ramp)
    w_all_frames = np.tile(np.abs(radius)[:, np.newaxis, np.newaxis], (1, SPOKES, T))

    # Dummy image (phantom)
    phantom = np.zeros((H, W, T), dtype=np.complex64)
    phantom[H//4:3*H//4, W//4:3*W//4, :] = 1.0
    for t in range(T):
        phantom[:,:,t] *= (1 - t/(2*T))
        
    # 2. Instantiate the MCNUFFT operator
    mcnufft_op = MCNUFFT(k=k_all_frames, w=w_all_frames, b1=smap_np, device='cpu')
    
    # 3. Test the forward operation
    print("\nTesting forward operation...")
    kspace_data = mcnufft_op.forward(phantom)
    print(f"Input image shape: {phantom.shape}")
    print(f"Output k-space shape: {kspace_data.shape}")
    expected_kspace_shape = (T, C, POINTS * SPOKES)
    print(f"Expected k-space shape: {expected_kspace_shape}")
    assert kspace_data.shape == expected_kspace_shape, "Forward shape mismatch!"
    print("Forward test PASSED.")

    # 4. Test the adjoint operation
    print("\nTesting adjoint operation...")
    reconstructed_image = mcnufft_op.adjoint(kspace_data)
    print(f"Input k-space shape: {kspace_data.shape}")
    print(f"Output image shape: {reconstructed_image.shape}")
    print(f"Expected image shape: {phantom.shape}")
    assert reconstructed_image.shape == phantom.shape, "Adjoint shape mismatch!"
    print("Adjoint test PASSED.")
    
    # 5. Visualize a result
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.abs(phantom[:,:,-1]), cmap='gray')
    axes[0].set_title('Original Image (Last Frame)')
    axes[1].imshow(np.abs(reconstructed_image.cpu().detach().numpy()[:,:,-1]), cmap='gray')
    axes[1].set_title('Adjoint NUFFT Recon (Last Frame)')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig("nufft_out.png")