import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import imageio # For creating GIFs
import bart

# Import the custom functions from your other Python files
from dro import gen_dro
from mask import disp_mask
from kspace import gen_kspace_data

# --- Main execution block for the full simulation pipeline ---
if __name__ == '__main__':

    # ===================================================================
    # --- 1. CONFIGURATION ---
    # ===================================================================
    # Path to the directory containing your .mat subject files
    # !!! PLEASE UPDATE THIS PATH if necessary !!!
    DATA_DIRECTORY = '/gpfs/data/karczmar-lab/workspaces/rachelgordon/DRO-BreastDCEMRI/data'
    
    # Range of subject files to load
    NUM_SUBJECTS = 53
    
    # Parameters for k-space generation
    SPOKES_PER_FRAME = 13
    NOISE_LEVEL = 0.05
    
    # Device for PyTorch computations ('cpu' or 'cuda:0' if you have a GPU)
    DEVICE = 'cpu'
    
    # --- Optional settings ---
    CREATE_DRO_GIF = True  # Set to False to skip GIF creation, which can be slow
    DISPLAY_MASKS = True   # Set to False to skip showing the mask plot

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
    dro_results = gen_dro(patient_data_list, option={}) # option={} is equivalent to MATLAB's []
    
    # Unpack the results for convenience
    simImg = dro_results['simImg']
    mask_dict = dro_results['mask']
    smap = dro_results['smap']
    S0 = dro_results['S0']
    
    print(f"DRO generation complete for case ID: {dro_results['ID']}")
    print(f"  - Simulated Image (simImg) shape: {simImg.shape}")

    # --- Optional: Create a GIF of the dynamic image series ---
    if CREATE_DRO_GIF:
        print("\nCreating GIF of the dynamic DRO image series...")
        frames = []
        # Set a consistent figure size and DPI for predictable output
        fig = plt.figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111)

        vmax = np.max(np.abs(simImg))
        for i in range(simImg.shape[2]):
            ax.clear() # Clear the axes for the new frame
            ax.imshow(np.abs(simImg[:, :, i]), cmap='gray', vmin=0, vmax=vmax)
            ax.set_title(f'Frame {i+1}/{simImg.shape[2]}')
            ax.axis('off')
            fig.tight_layout() # Adjust layout
            
            # This is the robust way to get the image data
            fig.canvas.draw()
            # Get the RGBA buffer and convert it to a NumPy array
            frame_img = np.array(fig.canvas.renderer.buffer_rgba())
            # We only need the RGB channels for the GIF
            frames.append(frame_img[:, :, :3])

        plt.close(fig)
        
        gif_filename = 'dro_simulation.gif'
        imageio.mimsave(gif_filename, frames, fps=5)
        print(f"Saved DRO animation to '{gif_filename}'")


    # ===================================================================
    # --- 4. DISPLAY MASKS ---
    # ===================================================================
    if DISPLAY_MASKS:
        print("\n--- Step 3: Displaying Segmentation Masks ---")
        # The disp_mask function will create and show a plot, then save it.
        _, _ = disp_mask(mask_dict, S0, show_plot=True)
        # Assuming disp_mask saves a file or you can modify it to do so.
        # For non-interactive use, you'd typically have it save a figure.
        print("Mask display generated.")


    # ===================================================================
    # --- 5. RADIAL K-SPACE DATA GENERATION ---
    # ===================================================================
    print("\n--- Step 4: Generating Radial K-space Data from DRO ---")
    kspace, traj, test_img = gen_kspace_data(
        simImg, smap, SPOKES_PER_FRAME, NOISE_LEVEL, device=DEVICE
    )
    
    # ===> ADD THIS FIX FROM EARLIER TO SCALE THE TRAJECTORY <===
    # This is still necessary for SigPy to work correctly.
    image_width = smap.shape[0]
    traj = traj * (image_width / 2)
    print(f"Trajectory scaled by a factor of {image_width / 2}.")
    # ===> END OF FIX <===

    print("K-space generation complete.")
    print(f"  - Final K-space shape: {kspace.shape}")
    print(f"  - Trajectory shape: {traj.shape}")
    print(f"  - Adjoint test image shape: {test_img.shape}")
    
    # Save a visualization of the k-space generation results
    fig_k, axes_k = plt.subplots(1, 2, figsize=(12, 6))
    axes_k[0].scatter(np.real(traj.flatten()), np.imag(traj.flatten()), s=0.1, alpha=0.5)
    axes_k[0].set_title('K-space Trajectory')
    axes_k[0].set_aspect('equal')
    axes_k[1].imshow(np.log(np.abs(kspace[:, :, 0]) + 1e-9), cmap='gray')
    axes_k[1].set_title('Log of K-space (Coil 0)')
    fig_k.suptitle('K-space Generation Results')
    plt.tight_layout()
    kspace_plot_filename = 'kspace_generation_results.png'
    plt.savefig(kspace_plot_filename)
    print(f"Saved k-space plot to '{kspace_plot_filename}'")
    plt.close(fig_k)




    # ===================================================================
    # --- 6. RADIAL RECONSTRUCTION USING SIGPY (CORRECTED) ---
    # ===================================================================
    print("\n--- Step 5: Performing Radial Reconstruction with SigPy ---")

    try:
        import sigpy as sp
        # ===> CORRECTED IMPORT STATEMENTS <===
        # SenseRecon is in the 'app' submodule.
        # pipe_menon_dcf is in the 'dcf' submodule.
        from sigpy.mri.app import SenseRecon
        from sigpy.mri.dcf import pipe_menon_dcf
    except ImportError:
        print("Error: The 'sigpy' library is not installed.")
        print("Please install it using: pip install sigpy")
        exit()
    except AttributeError:
        print("Error: A function was not found in sigpy. Your sigpy version might be old.")
        print("Please try updating it: pip install --upgrade sigpy")
        exit()

    # --- 6.1. Prepare data dimensions for SigPy ---
    spokes_per_frame = SPOKES_PER_FRAME
    nt = kspace.shape[1] // spokes_per_frame
    kspace_trim = kspace[:, :spokes_per_frame * nt, :]
    traj_trim = traj[:, :spokes_per_frame * nt]

    nx_readout, ntview, ncoil = kspace_trim.shape

    # Sensitivity maps need to be (num_coils, ny, nx)
    smap_sigpy = np.ascontiguousarray(smap.transpose(2, 0, 1))

    reco_frames = []
    print(f"Starting frame-by-frame reconstruction for {nt} frames...")

    for t in range(nt):
        print(f"  Reconstructing frame {t + 1}/{nt}...")

        # --- Select the data for the current time frame ---
        start_idx = t * spokes_per_frame
        end_idx = (t + 1) * spokes_per_frame
        
        # K-space data: (coils, spokes * readout)
        ksp_frame_t = kspace_trim[:, start_idx:end_idx, :].transpose(2, 1, 0).reshape(ncoil, -1)
        
        # Trajectory coordinates: (spokes * readout, num_dims) e.g., (..., 2) for 2D
        traj_frame_t = traj_trim[:, start_idx:end_idx]
        coords_frame_t = np.stack((np.real(traj_frame_t), np.imag(traj_frame_t)), axis=-1).reshape(-1, 2)
        
        # --- Calculate Density Compensation Factors (DCF) ---
        dcf_weights = pipe_menon_dcf(coords_frame_t, img_shape=smap_sigpy.shape[1:])
        
        # --- Perform the reconstruction using the correct high-level application ---
        reco_frame_t = SenseRecon(
            ksp_frame_t,
            mps=smap_sigpy,
            coord=coords_frame_t,
            #weights=dcf_weights,
            max_iter=20
        ).run()
        
        reco_frames.append(reco_frame_t)

    # --- 6.2. Finalize and Visualize ---
    reco_sigpy = np.stack(reco_frames, axis=-1)

    print("\nSigPy reconstruction complete.")
    print(f"  - Reconstructed image shape: {reco_sigpy.shape}")

    # --- 6.3. Visualize and save the reconstruction ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(np.abs(simImg[:, :, 0]), cmap='gray')
    axes[0].set_title('Original DRO (Frame 1)')
    axes[0].axis('off')

    axes[1].imshow(np.abs(reco_sigpy[:, :, 0]), cmap='gray')
    axes[1].set_title('SigPy Reconstructed Image (Frame 1)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('reconstruction_comparison_sigpy.png')
    print("Saved reconstruction comparison to 'reconstruction_comparison_sigpy.png'")

    print("\nCreating GIF of the SigPy reconstructed image series...")
    frames_reco = [np.abs(reco_sigpy[:, :, i]) for i in range(reco_sigpy.shape[2])]
    vmax_reco = np.max(frames_reco) if frames_reco else 1.0
    if vmax_reco > 0:
        frames_reco_uint8 = [(255 * f / vmax_reco).astype(np.uint8) for f in frames_reco]
        imageio.mimsave('reconstruction_sigpy.gif', frames_reco_uint8, fps=5, loop=0)
        print("Saved SigPy reconstruction animation to 'reconstruction_sigpy.gif'")
    else:
        print("Skipping reconstruction GIF as the output is all zeros.")


    # # ===================================================================
    # # --- 5. DATA GENERATION SANITY CHECK (CORRECTED) ---
    # # ===================================================================
    # print("\n--- Step 4: Sanity Check - Generating 'Perfect' Data with SigPy ---")
    # print("Replacing call to custom 'gen_kspace_data' function.")

    # try:
    #     import sigpy as sp
    #     from sigpy.mri import radial
    # except ImportError:
    #     print("Error: The 'sigpy' library is not installed.")
    #     print("Please install it using: pip install sigpy")
    #     exit()

    # # --- Setup for SigPy-based simulation ---
    # smap_sigpy = np.ascontiguousarray(smap.transpose(2, 0, 1))
    # ncoil, image_height, image_width = smap_sigpy.shape
    # img_frame = simImg[:, :, 0] 

    # # --- Generate a known-good radial trajectory ---
    # # The radial function generates coordinates correctly scaled between -width/2 and +width/2.
    # coords = radial(
    #     coord_shape=(SPOKES_PER_FRAME, image_width, 2),
    #     img_shape=(image_height, image_width)
    # ).reshape(-1, 2)

    # # --- Simulate the k-space acquisition using a forward NUFFT ---
    # # 1. Create the NUFFT operator (multi-coil image -> multi-coil k-space)
    # F = sp.linop.NUFFT(smap_sigpy.shape, coords)

    # # 2. Create the Sensitivity operator (single-coil image -> multi-coil image)
    # #    ===> THE FIX IS HERE <===
    # #    The input shape of S must be the shape of the image we are feeding it (img_frame).
    # S = sp.linop.Multiply(img_frame.shape, smap_sigpy)

    # # 3. Chain them together: A = F * S
    # #    The full operator A now correctly represents:
    # #    single-channel image -> multi-channel image -> multi-channel k-space
    # A = F * S

    # # 4. Apply the forward model to the ground truth image to get perfect k-space
    # print("Generating known-good k-space data...")
    # kspace = A(img_frame)

    # # 5. Add a small amount of noise
    # noise = NOISE_LEVEL * np.random.standard_normal(kspace.shape).astype(np.complex64)
    # kspace += noise

    # print("Known-good data generation complete.")


    # # ===================================================================
    # # --- 6. ADJOINT RECONSTRUCTION TEST (on known-good data) ---
    # # ===================================================================
    # print("\n--- Step 5: Performing Adjoint Reconstruction on Known-Good Data ---")

    # # The adjoint of the forward operator A is A.H
    # print("Performing Adjoint Reconstruction (A.H * k)...")
    # adjoint_reco = A.H(kspace)

    # print("Adjoint reconstruction test complete.")

    # # --- Visualize the result ---
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # axes[0].imshow(np.abs(img_frame), cmap='gray')
    # axes[0].set_title('Original DRO (Frame 1)')
    # axes[0].axis('off')

    # axes[1].imshow(np.abs(adjoint_reco), cmap='gray')
    # axes[1].set_title('Adjoint Recon of SigPy-Generated Data')
    # axes[1].axis('off')

    # plt.tight_layout()
    # plt.savefig('reconstruction_sanity_check.png')
    # print("Saved sanity check image to 'reconstruction_sanity_check.png'")



    # # ===================================================================
    # # --- 6. RADIAL RECONSTRUCTION USING BART (NEW SECTION) ---
    # # ===================================================================
    # print("\n--- Step 5: Performing Radial Reconstruction with BART ---")

    # # --- 6.1. Prepare data dimensions for BART ---
    
    # # Trim k-space and trajectory to have an integer number of frames
    # spokes_per_frame = SPOKES_PER_FRAME
    # nt = kspace.shape[1] // spokes_per_frame
    # kspace_trim = kspace[:, :spokes_per_frame * nt, :]
    # traj_trim = traj[:, :spokes_per_frame * nt]

    # nx, ntview, ncoil = kspace_trim.shape
    
    # # Reshape k-space for BART
    # # MATLAB: reshape(kspace_trim,[1,nx,spokes_per_frame,nt,ncoil]);
    # # MATLAB: permute(...,[1,2,3,5,4]); -> (1, nx, spokes, ncoil, nt)
    # # MATLAB: reshape(...,[1,nx,spokes_per_frame,ncoil,1,1,1,1,1,1,nt]);
    # # This places data in specific dimensions for BART:
    # # (1, readout, spokes, coils, map, ..., time)
    # kspace_reshaped = kspace_trim.reshape((1, nx, spokes_per_frame, nt, ncoil), order='F')
    # kspace_permuted = kspace_reshaped.transpose(0, 1, 2, 4, 3)
    # final_ksp_shape = (1, nx, spokes_per_frame, ncoil, 1, 1, 1, 1, 1, 1, nt)
    # kspace_dim = kspace_permuted.reshape(final_ksp_shape)
    
    # print(f"Reshaped k-space for BART to shape: {kspace_dim.shape}")

    # # Reshape trajectory for BART
    # # Create 3-channel trajectory (x, y, z) and scale it
    # traj_dim_stack = np.stack([
    #     np.real(traj_trim),
    #     np.imag(traj_trim),
    #     np.zeros_like(np.real(traj_trim))
    # ]) * (nx / 2)
    
    # # Reshape to match k-space dimensions (readout, spokes, time)
    # # The 'F' order is crucial to split ntview -> (spokes, time) correctly
    # traj_reshaped = traj_dim_stack.reshape((3, nx, spokes_per_frame, nt), order='F')
    
    # # Expand to the final 11D BART format
    # final_traj_shape = (3, nx, spokes_per_frame, 1, 1, 1, 1, 1, 1, 1, nt)
    # traj_dim = traj_reshaped.reshape(final_traj_shape)

    # print(f"Reshaped trajectory for BART to shape: {traj_dim.shape}")

    # # Reshape sensitivity maps for BART
    # # BART pics expects smaps in (x, y, z, coils) format.
    # smap_dim = smap[:, :, np.newaxis, :]
    # print(f"Reshaped sensitivity maps for BART to shape: {smap_dim.shape}")

    # # --- 6.2. Run BART PICS reconstruction ---
    # print("Executing BART command: pics -S -RT:1024:0:0.01 -i100 -t ...")
    
    # # The `bart` function takes (num_outputs, command_str, input_array1, ...)
    # reco = bart.bart(1, 'pics -S -RT:1024:0:0.01 -i100 -t', traj_dim, kspace_dim, smap_dim)
    
    # # Process the output: take absolute value and remove singleton dimensions
    # grasp_bart = np.squeeze(np.abs(reco))
    
    # print("BART reconstruction complete.")
    # print(f"  - Reconstructed image shape: {grasp_bart.shape}")

    # # --- 6.3. Visualize and save the reconstruction ---
    # # Compare the original DRO with the reconstructed image
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # # Original DRO (first time frame)
    # axes[0].imshow(np.abs(simImg[:, :, 0]), cmap='gray')
    # axes[0].set_title('Original DRO (Frame 1)')
    # axes[0].axis('off')
    
    # # Reconstructed Image (first time frame)
    # axes[1].imshow(grasp_bart[:, :, 0], cmap='gray')
    # axes[1].set_title('BART Reconstructed Image (Frame 1)')
    # axes[1].axis('off')
    
    # plt.tight_layout()
    # plt.savefig('reconstruction_comparison.png')
    # print("Saved reconstruction comparison to 'reconstruction_comparison.png'")
    # # plt.show() # Uncomment to display the plot interactively

    # # Create a GIF of the reconstructed dynamic series
    # print("\nCreating GIF of the BART reconstructed image series...")
    # frames_reco = [grasp_bart[:, :, i] for i in range(grasp_bart.shape[2])]
    # vmax_reco = np.max(frames_reco)
    # if vmax_reco > 0:
    #     frames_reco_uint8 = [(255 * f / vmax_reco).astype(np.uint8) for f in frames_reco]
    #     imageio.mimsave('reconstruction_bart.gif', frames_reco_uint8, fps=5, loop=0)
    #     print("Saved BART reconstruction animation to 'reconstruction_bart.gif'")
    # else:
    #     print("Skipping reconstruction GIF as the output is all zeros.")