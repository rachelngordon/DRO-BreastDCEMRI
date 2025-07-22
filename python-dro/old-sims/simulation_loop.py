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


### MODIFICATION ###: The function now accepts a single patient data dictionary.
def run_simulation(case_num, single_patient_data, config):
    """
    Runs a single, full simulation pipeline for one specific patient case.
    
    This function generates a DRO based on the provided patient data, simulates k-space,
    reconstructs the image, and saves all relevant data to a structured directory.

    Args:
        case_num (int): The number for the output directory (e.g., 1, 2, 3...).
        single_patient_data (dict): A dictionary for ONE patient, containing S0, smap, etc.
        config (dict): A dictionary containing all configuration parameters.
    """
    # ===================================================================
    # --- A. SETUP DIRECTORIES FOR THIS CASE ---
    # ===================================================================
    # The output folder is named based on the case number (DRO_001, DRO_002, etc.)
    case_id_str = f"DRO_{case_num:03d}"
    sample_dir = os.path.join(config['OUTPUT_DIRECTORY'], case_id_str)
    dro_gt_dir = os.path.join(sample_dir, 'DRO')
    recon_dir = os.path.join(sample_dir, 'recon')

    os.makedirs(dro_gt_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    
    # Print which real patient ID is being used for this simulation case
    underlying_patient_id = single_patient_data['ID']
    print(f"\n--- Processing Simulation Case: {case_id_str} (based on real patient: {underlying_patient_id}) ---")
    print(f"Saving results to: {os.path.abspath(sample_dir)}")

    # ===================================================================
    # --- B. GENERATE DIGITAL REFERENCE OBJECT (DRO) ---
    # ===================================================================
    print("Step 1: Generating DRO from the specified patient case...")
    
    ### MODIFICATION ###: Pass a list containing only the single patient's data to gen_dro.
    # The [ ] are important as gen_dro likely expects a list.
    dro_results = gen_dro([single_patient_data], option={}) 
    
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
    S = sp.linop.Multiply(smap_sigpy.shape[1:], smap_sigpy)
    A = F * S

    # --- Loop Through Frames, Simulate, and Reconstruct ---
    reco_frames = []
    kspace_frames = [] 
    
    print(f"Starting frame-by-frame simulation for {num_frames} frames...")
    for t in range(num_frames):
        img_frame_t = simImg[:, :, t]
        kspace_frame_t = A(img_frame_t)
        
        noise = config['NOISE_LEVEL'] * (
            np.random.standard_normal(kspace_frame_t.shape) + 
            1j * np.random.standard_normal(kspace_frame_t.shape)
        ).astype(np.complex64)
        kspace_frame_t += noise
        kspace_frames.append(kspace_frame_t)
        
        reco_frame_t = SenseRecon(
            kspace_frame_t,
            mps=smap_sigpy,
            coord=coords,
            max_iter=15
        ).run()
        reco_frames.append(reco_frame_t)

    # --- Finalize Tensors ---
    reco_final = np.stack(reco_frames, axis=-1)
    kspace_full = np.stack(kspace_frames, axis=0)
    
    print("SigPy pipeline complete.")

    # ===================================================================
    # --- D. SAVE ALL SIMULATED DATA TO DISK ---
    # ===================================================================
    print("Step 3: Saving all data files to disk...")

    # Save the ID of the source patient for traceability
    with open(os.path.join(sample_dir, 'source_patient_id.txt'), 'w') as f:
        f.write(underlying_patient_id)
        
    np.save(os.path.join(sample_dir, 'coil_sensitivity_maps.npy'), smap)
    np.save(os.path.join(sample_dir, 'kspace_trajectory.npy'), coords)
    np.save(os.path.join(sample_dir, 'simulated_kspace.npy'), kspace_full)

    affine = np.eye(4)
    for t in range(num_frames):
        gt_frame_abs = np.abs(simImg[:, :, t])
        gt_nifti = nib.Nifti1Image(gt_frame_abs, affine)
        gt_filename = os.path.join(dro_gt_dir, f"frame_{t+1:03d}.nii")
        nib.save(gt_nifti, gt_filename)

        reco_frame_abs = np.abs(reco_final[:, :, t])
        reco_nifti = nib.Nifti1Image(reco_frame_abs, affine)
        reco_filename = os.path.join(recon_dir, f"frame_{t+1:03d}.nii")
        nib.save(reco_nifti, reco_filename)

    print("Successfully saved all .npy and .nii files.")

    # ... (Optional visualization saving block E remains the same) ...

    print(f"--- Finished processing case: {case_id_str} ---")


# --- Main execution block for the full simulation pipeline ---
if __name__ == '__main__':
    
    start_time = time.time()

    # ===================================================================
    # --- 1. CONFIGURATION ---
    # ===================================================================
    CONFIG = {
        'DATA_DIRECTORY': '/gpfs/data/karczmar-lab/workspaces/rachelgordon/DRO-BreastDCEMRI/data',
        'OUTPUT_DIRECTORY': '/ess/scratch/scratch1/rachelgordon/simulated_dataset', 
        # 'NUM_CASES_TO_SIMULATE' is no longer needed, as we loop over all loaded subjects.
        'NUM_SUBJECTS_TO_LOAD': 53,
        'SPOKES_PER_FRAME': 36,
        'NOISE_LEVEL': 0.05,
        'DEVICE': 'cpu',
        'SAVE_PLOTS_AND_GIFS': False # Set to False for faster batch generation
    }
    
    os.makedirs(CONFIG['OUTPUT_DIRECTORY'], exist_ok=True)

    # ===================================================================
    # --- 2. DATA COLLECTION (from .mat files) ---
    # ===================================================================
    print("--- Step 1: Loading All Real Patient Data ---")
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

    num_patients_loaded = len(patient_data_list)
    print(f"Successfully loaded {num_patients_loaded} patient cases.\n")
    
    # ===================================================================
    # --- 3. RUN SIMULATION LOOP ---
    # ===================================================================
    print(f"--- Starting Batch Simulation for all {num_patients_loaded} Loaded Patients ---")
    
    ### MODIFICATION ###: Loop directly over the loaded patient data list.
    # 'enumerate(..., start=1)' gives us a counter 'i' that starts at 1.
    for i, single_patient_data in enumerate(patient_data_list, start=1):
        try:
            # Pass the case number (i) and the specific patient's data to the function.
            run_simulation(
                case_num=i,
                single_patient_data=single_patient_data,
                config=CONFIG
            )
        except Exception as e:
            patient_id = single_patient_data.get('ID', f'Case {i}')
            print(f"\n!!!!!! ERROR processing {patient_id} !!!!!!!")
            print(f"Error details: {e}")
            print("Skipping to the next case.")
            import traceback
            traceback.print_exc()
            continue

    end_time = time.time()
    total_time = end_time - start_time
    print("\n=========================================================")
    print("          BATCH SIMULATION COMPLETE")
    print(f"  Total cases simulated: {num_patients_loaded}")
    print(f"  Total time elapsed: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"  Data saved in: {os.path.abspath(CONFIG['OUTPUT_DIRECTORY'])}")
    print("=========================================================")