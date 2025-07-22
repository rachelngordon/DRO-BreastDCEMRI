import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import imageio # For creating GIFs

# Import the custom functions from your other Python files
from dro import gen_dro
from mask import disp_mask
from kspace import gen_kspace_data
import nibabel as nib

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
    dro_results = gen_dro(patient_data_list, option={}) 
    simImg = dro_results['simImg']
    mask_dict = dro_results['mask']
    smap = dro_results['smap']
    S0 = dro_results['S0']
    print(f"DRO generation complete for case ID: {dro_results['ID']}")
    print(f"  - Simulated Image (simImg) shape: {simImg.shape}")



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
    # --- 5. DATA GENERATION & RECONSTRUCTION (SIGPY PIPELINE) ---
    # ===================================================================
    print("\n--- Step 4: Using a Full SigPy Pipeline for Simulation & Reconstruction ---")

    try:
        import sigpy as sp
        from sigpy.mri import radial
        from sigpy.mri.app import SenseRecon
    except ImportError:
        print("Error: The 'sigpy' library is not installed.")
        print("Please install it using: pip install sigpy")
        exit()

    # --- 5.1. Prepare Data Dimensions ---
    smap_sigpy = np.ascontiguousarray(smap.transpose(2, 0, 1))
    ncoil, image_height, image_width = smap_sigpy.shape

    # --- 5.2. Generate Known-Good Trajectory ---
    coords = radial(
        coord_shape=(SPOKES_PER_FRAME, image_width, 2),
        img_shape=(image_height, image_width)
    ).reshape(-1, 2)

    # --- 5.3. Build the SENSE Forward Model (A) ---
    F = sp.linop.NUFFT(smap_sigpy.shape, coords)
    S = sp.linop.Multiply(simImg[:, :, 0].shape, smap_sigpy)
    A = F * S

    # --- 5.4. Loop Through Frames, Simulate, and Reconstruct ---
    reco_frames = []
    # ===> ADDITION: Create a list to store k-space frames <===
    kspace_frames = [] 
    
    num_frames = simImg.shape[2]
    print(f"Starting frame-by-frame simulation and reconstruction for {num_frames} frames...")

    for t in range(num_frames):
        print(f"  Processing frame {t + 1}/{num_frames}...")
        
        img_frame_t = simImg[:, :, t]
        kspace_frame_t = A(img_frame_t)
        noise = NOISE_LEVEL * np.random.standard_normal(kspace_frame_t.shape).astype(np.complex64)
        kspace_frame_t += noise
        
        # ===> ADDITION: Append the generated k-space frame to our list <===
        kspace_frames.append(kspace_frame_t)
        
        reco_frame_t = SenseRecon(
            kspace_frame_t,
            mps=smap_sigpy,
            coord=coords,
            max_iter=15
        ).run()
        
        reco_frames.append(reco_frame_t)

    # --- 5.5. Finalize Tensors ---
    reco_final = np.stack(reco_frames, axis=-1)
    # ===> ADDITION: Stack all k-space frames into one large tensor <===
    kspace_full = np.stack(kspace_frames, axis=0)
    
    print("\nFull SigPy pipeline complete.")
    print(f"  - Final reconstructed image shape: {reco_final.shape}")

    # ===================================================================
    # --- 5.6. SAVE SIMULATED DATA (NEW SECTION) ---
    # ===================================================================
    print("\n--- Saving Simulated Data to Disk ---")
    
    # Print the final shapes of the data
    print(f"  - Full K-space Data Shape: {kspace_full.shape} (Frames, Coils, Samples)")
    print(f"  - K-space Trajectory Shape: {coords.shape} (Samples, Dims)")
    print(f"  - Coil Sensitivity Maps Shape: {smap.shape} (H, W, Coils)")

    # Save the arrays to .npy files
    np.save('simulated_kspace.npy', kspace_full)
    np.save('kspace_trajectory.npy', coords)
    np.save('coil_sensitivity_maps.npy', smap)

    print("\nSuccessfully saved:")
    print("  - 'simulated_kspace.npy'")
    print("  - 'kspace_trajectory.npy'")
    print("  - 'coil_sensitivity_maps.npy'")
    # ===================================================================


    # --- 5.7. Finalize and Visualize --- (Renumbered from 5.5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(np.abs(simImg[:, :, 0]), cmap='gray')
    axes[0].set_title('Original DRO (Frame 1)')
    axes[1].imshow(np.abs(reco_final[:, :, 0]), cmap='gray')
    axes[1].set_title('Final SigPy Reconstructed Image (Frame 1)')
    plt.tight_layout()
    plt.savefig('reconstruction_final_sigpy.png')
    print("\nSaved final reconstruction image to 'reconstruction_final_sigpy.png'")

    print("\nCreating GIF of the final reconstructed image series...")
    frames_reco = [np.abs(reco_final[:, :, i]) for i in range(reco_final.shape[2])]
    vmax_reco = np.max(frames_reco) if frames_reco else 1.0
    if vmax_reco > 0:
        frames_reco_uint8 = [(255 * f / vmax_reco).astype(np.uint8) for f in frames_reco]
        imageio.mimsave('reconstruction_final_sigpy.gif', frames_reco_uint8, fps=5, loop=0)
        print("Saved final animation to 'reconstruction_final_sigpy.gif'")
    else:
        print("Skipping reconstruction GIF as the output is all zeros.")


    # ===================================================================
    # --- 5.7. SAVE GROUND TRUTH AND RECONSTRUCTED IMAGES ---
    # ===================================================================
    if nib:
        print("\nSaving ground truth and reconstructed frames as NIFTI files...")
        num_frames_to_save = simImg.shape[2]
        affine = np.eye(4) # Default identity affine

        # Loop to save both sets of images
        for t in range(num_frames_to_save):
            # --- Save the Absolute Ground Truth (simImg) ---
            abs_gt_frame = np.abs(simImg[:, :, t])
            abs_gt_nifti = nib.Nifti1Image(abs_gt_frame, affine)
            abs_gt_filename = f"sim_abs_gt_frame_{t+1:03d}.nii"
            nib.save(abs_gt_nifti, abs_gt_filename)

            # --- Save the Reconstructed Baseline/Benchmark (reco_final) ---
            reco_gt_frame = np.abs(reco_final[:, :, t])
            reco_gt_nifti = nib.Nifti1Image(reco_gt_frame, affine)
            reco_gt_filename = f"sim_reco_gt_frame_{t+1:03d}.nii"
            nib.save(reco_gt_nifti, reco_gt_filename)

        print(f"Successfully saved {num_frames_to_save} frames for both types:")
        print("  - Absolute Ground Truth (e.g., 'sim_abs_gt_frame_001.nii')")
        print("  - Reconstructed Baseline (e.g., 'sim_reco_gt_frame_001.nii')")



    # ===================================================================
    # --- 6. PLOT ALL TIME-FRAMES IN A GRID ---
    # ===================================================================
    print("\nGenerating comparison plots for all timeframes...")

    # --- Plot 1: Original DRO (Ground Truth) ---
    num_frames = simImg.shape[2]
    # Dynamically determine the grid size for the plot
    plot_cols = int(np.ceil(np.sqrt(num_frames)))
    plot_rows = int(np.ceil(num_frames / plot_cols))
    
    fig1, axes1 = plt.subplots(plot_rows, plot_cols, figsize=(plot_cols * 3, plot_rows * 3))
    fig1.suptitle('Original DRO (Ground Truth) - All Frames', fontsize=20)
    
    # Flatten the axes array for easy iteration
    axes1 = axes1.ravel() 
    
    vmax_dro = np.max(np.abs(simImg)) # Consistent brightness
    for i in range(num_frames):
        axes1[i].imshow(np.abs(simImg[:, :, i]), cmap='gray', vmin=0, vmax=vmax_dro)
        axes1[i].set_title(f'Frame {i+1}')
        axes1[i].axis('off')
        
    # Hide any unused subplots
    for i in range(num_frames, len(axes1)):
        axes1[i].axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.savefig('original_dro_all_frames.png')
    plt.close(fig1)
    print("Saved plot of all original DRO frames to 'original_dro_all_frames.png'")


    # --- Plot 2: Final Reconstructed Images ---
    fig2, axes2 = plt.subplots(plot_rows, plot_cols, figsize=(plot_cols * 3, plot_rows * 3))
    fig2.suptitle('Final SigPy Reconstruction - All Frames', fontsize=20)
    
    # Flatten the axes array
    axes2 = axes2.ravel()
    
    vmax_reco_all = np.max(np.abs(reco_final)) # Consistent brightness
    for i in range(num_frames):
        axes2[i].imshow(np.abs(reco_final[:, :, i]), cmap='gray', vmin=0, vmax=vmax_reco_all)
        axes2[i].set_title(f'Frame {i+1}')
        axes2[i].axis('off')
        
    # Hide any unused subplots
    for i in range(num_frames, len(axes2)):
        axes2[i].axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.savefig('reconstruction_all_frames.png')
    plt.close(fig2)
    print("Saved plot of all reconstructed frames to 'reconstruction_all_frames.png'")
