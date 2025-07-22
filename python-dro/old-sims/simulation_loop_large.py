import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import imageio
import nibabel as nib
import time

# Import the custom functions from your other Python files
# Make sure these files are in the same directory or in your Python path
from dro import gen_dro
from mask import disp_mask
from kspace import gen_kspace_data

# It's good practice to wrap the SigPy import in a try-except block
try:
    import sigpy as sp
    from sigpy.mri import radial
    from sigpy.mri.app import SenseRecon
    SIGPY_AVAILABLE = True
except ImportError:
    print("Warning: The 'sigpy' library is not installed or could not be imported.")
    print("The simulation and reconstruction pipeline will not run.")
    print("Please install it using: pip install sigpy")
    SIGPY_AVAILABLE = False


def run_simulation(case_num, patient_data_list, config):
    """
    Runs a single, full simulation pipeline for one case.
    
    This function generates a DRO, simulates k-space, reconstructs the image,
    and saves all relevant data to a structured directory.

    Args:
        case_num (int): The number of the current simulation case (e.g., 1, 2, 3...).
        patient_data_list (list): A list of dictionaries containing the real patient data.
        config (dict): A dictionary containing all configuration parameters.
    """
    # ===================================================================
    # --- A. SETUP DIRECTORIES FOR THIS CASE ---
    # ===================================================================
    case_id_str = f"DRO_{case_num:03d}"
    sample_dir = os.path.join(config['OUTPUT_DIRECTORY'], case_id_str)
    dro_gt_dir = os.path.join(sample_dir, 'DRO')
    recon_dir = os.path.join(sample_dir, 'recon')

    # Create all necessary directories
    os.makedirs(dro_gt_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    
    print(f"\n--- Processing Case: {case_id_str} ---")
    print(f"Saving results to: {os.path.abspath(sample_dir)}")

    # ===================================================================
    # --- B. GENERATE DIGITAL REFERENCE OBJECT (DRO) ---
    # ===================================================================
    print("Step 1: Generating DRO from a randomly selected patient case...")
    dro_results = gen_dro(patient_data_list, option={}) 
    simImg = dro_results['simImg']
    mask_dict = dro_results['mask']
    smap = dro_results['smap']
    S0 = dro_results['S0']
    
    print(f"DRO generation complete for underlying patient ID: {dro_results['ID']}")
    print(f"  - Simulated Image (simImg) shape: {simImg.shape}")

    # ===================================================================
    # --- C. DATA GENERATION & RECONSTRUCTION (SIGPY PIPELINE) ---
    # ===================================================================
    if not SIGPY_AVAILABLE:
        print("Skipping simulation and reconstruction because SigPy is not available.")
        return

    print("Step 2: Using a SigPy pipeline for simulation & reconstruction...")

    # --- Prepare Data Dimensions ---
    smap_sigpy = np.ascontiguousarray(smap.transpose(2, 0, 1))
    ncoil, image_height, image_width = smap_sigpy.shape
    num_frames = simImg.shape[2]

    # --- Generate K-space Trajectory ---
    coords = radial(
        coord_shape=(config['SPOKES_PER_FRAME'], image_width, 2),
        img_shape=(image_height, image_width)
    ).reshape(-1, 2)

    # --- Build the SENSE Forward Model (A) ---
    F = sp.linop.NUFFT(smap_sigpy.shape, coords)
    S = sp.linop.Multiply(smap_sigpy.shape[1:], smap_sigpy) # Multiply op shape is (H, W)
    A = F * S

    # --- Loop Through Frames, Simulate, and Reconstruct ---
    reco_frames = []
    kspace_frames = [] 
    
    print(f"Starting frame-by-frame simulation for {num_frames} frames...")
    for t in range(num_frames):
        img_frame_t = simImg[:, :, t]
        
        # Simulate k-space data for this frame
        kspace_frame_t = A(img_frame_t)
        
        # Add complex Gaussian noise
        noise = config['NOISE_LEVEL'] * (
            np.random.standard_normal(kspace_frame_t.shape) + 
            1j * np.random.standard_normal(kspace_frame_t.shape)
        ).astype(np.complex64)
        kspace_frame_t += noise
        kspace_frames.append(kspace_frame_t)
        
        # Reconstruct the image from noisy k-space
        reco_frame_t = SenseRecon(
            kspace_frame_t,
            mps=smap_sigpy,
            coord=coords,
            max_iter=15
        ).run()
        reco_frames.append(reco_frame_t)

    # --- Finalize Tensors ---
    reco_final = np.stack(reco_frames, axis=-1)
    kspace_full = np.stack(kspace_frames, axis=0) # Shape: (frames, coils, samples)
    
    print("SigPy pipeline complete.")

    # ===================================================================
    # --- D. SAVE ALL SIMULATED DATA TO DISK ---
    # ===================================================================
    print("Step 3: Saving all data files to disk...")

    # --- Save primary files (.npy) ---
    np.save(os.path.join(sample_dir, 'coil_sensitivity_maps.npy'), smap)
    np.save(os.path.join(sample_dir, 'kspace_trajectory.npy'), coords)
    np.save(os.path.join(sample_dir, 'simulated_kspace.npy'), kspace_full)

    # --- Save ground truth and reconstructed frames (.nii) ---
    affine = np.eye(4) # Use a default identity affine matrix
    for t in range(num_frames):
        # Save absolute ground truth (from simImg) to the 'DRO' subdirectory
        gt_frame_abs = np.abs(simImg[:, :, t])
        gt_nifti = nib.Nifti1Image(gt_frame_abs, affine)
        gt_filename = os.path.join(dro_gt_dir, f"frame_{t+1:03d}.nii")
        nib.save(gt_nifti, gt_filename)

        # Save reconstructed image to the 'recon' subdirectory
        reco_frame_abs = np.abs(reco_final[:, :, t])
        reco_nifti = nib.Nifti1Image(reco_frame_abs, affine)
        reco_filename = os.path.join(recon_dir, f"frame_{t+1:03d}.nii")
        nib.save(reco_nifti, reco_filename)

    print("Successfully saved all .npy and .nii files.")

    # ===================================================================
    # --- E. SAVE VISUALIZATIONS (optional) ---
    # ===================================================================
    if config.get('SAVE_PLOTS_AND_GIFS', False):
        print("Step 4: Generating and saving visualization plots and GIFs...")
        
        # --- Save comparison of first frame ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(np.abs(simImg[:, :, 0]), cmap='gray')
        axes[0].set_title('Ground Truth DRO (Frame 1)')
        axes[1].imshow(np.abs(reco_final[:, :, 0]), cmap='gray')
        axes[1].set_title('SENSE Recon (Frame 1)')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, 'comparison_frame1.png'))
        plt.close(fig)

        # --- Create and save DRO GIF ---
        frames_dro = [np.abs(simImg[:, :, i]) for i in range(num_frames)]
        vmax_dro = np.max(frames_dro) if frames_dro else 1.0
        frames_dro_uint8 = [(255 * f / vmax_dro).astype(np.uint8) for f in frames_dro]
        imageio.mimsave(os.path.join(sample_dir, 'ground_truth_dro.gif'), frames_dro_uint8, fps=5, loop=0)

        # --- Create and save Recon GIF ---
        frames_reco = [np.abs(reco_final[:, :, i]) for i in range(num_frames)]
        vmax_reco = np.max(frames_reco) if frames_reco else 1.0
        if vmax_reco > 0:
            frames_reco_uint8 = [(255 * f / vmax_reco).astype(np.uint8) for f in frames_reco]
            imageio.mimsave(os.path.join(sample_dir, 'reconstruction.gif'), frames_reco_uint8, fps=5, loop=0)
        
        print("Visualizations saved.")

    print(f"--- Finished processing case: {case_id_str} ---")


# --- Main execution block for the full simulation pipeline ---
if __name__ == '__main__':
    
    start_time = time.time()

    # ===================================================================
    # --- 1. CONFIGURATION ---
    # ===================================================================
    ### MODIFICATION ###: Define number of cases and output directory
    CONFIG = {
        'DATA_DIRECTORY': '/gpfs/data/karczmar-lab/workspaces/rachelgordon/DRO-BreastDCEMRI/data',
        'OUTPUT_DIRECTORY': '/ess/scratch/scratch1/rachelgordon/simulated_dataset', # Main output folder
        'NUM_CASES_TO_SIMULATE': 100, # Example: set this to 50-100 as needed
        'NUM_SUBJECTS_TO_LOAD': 53,
        'SPOKES_PER_FRAME': 36,
        'NOISE_LEVEL': 0.05,
        'DEVICE': 'cpu', # Note: SigPy primarily uses CPU for these ops
        'SAVE_PLOTS_AND_GIFS': True # Set to False to speed up generation
    }
    
    # Create the main output directory if it doesn't exist
    os.makedirs(CONFIG['OUTPUT_DIRECTORY'], exist_ok=True)

    # ===================================================================
    # --- 2. DATA COLLECTION (from .mat files) ---
    # ===================================================================
    print("--- Step 1: Loading Real Patient Data (occurs once) ---")
    patient_data_list = []
    print(f"Attempting to load data from: {os.path.abspath(CONFIG['DATA_DIRECTORY'])}")

    for i in range(1, CONFIG['NUM_SUBJECTS_TO_LOAD'] + 1):
        filename = f'sub{i}.mat'
        full_path = os.path.join(CONFIG['DATA_DIRECTORY'], filename)
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
    # --- 3. RUN SIMULATION LOOP ---
    # ===================================================================
    print(f"--- Starting Batch Simulation for {CONFIG['NUM_CASES_TO_SIMULATE']} Cases ---")
    
    for i in range(1, CONFIG['NUM_CASES_TO_SIMULATE'] + 1):
        try:
            # ### MODIFICATION ###: Call the main simulation function for each case
            run_simulation(
                case_num=i,
                patient_data_list=patient_data_list,
                config=CONFIG
            )
        except Exception as e:
            print(f"\n!!!!!! ERROR processing case {i} !!!!!!!")
            print(f"Error details: {e}")
            print("Skipping to the next case.")
            # For debugging, you might want to uncomment the following line:
            # import traceback; traceback.print_exc()
            continue

    end_time = time.time()
    total_time = end_time - start_time
    print("\n=========================================================")
    print("          BATCH SIMULATION COMPLETE")
    print(f"  Total cases simulated: {CONFIG['NUM_CASES_TO_SIMULATE']}")
    print(f"  Total time elapsed: {total_time:.2f} seconds")
    print(f"  Data saved in: {os.path.abspath(CONFIG['OUTPUT_DIRECTORY'])}")
    print("=========================================================")