import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat  # Import the .mat file loader
from dro_resample_nonuniform import gen_dro      # Import your main function

# --- Main execution block for using real data ---
if __name__ == '__main__':
    print("--- Running Digital Reference Object (DRO) Generation with REAL data ---")

    #
    # <<< STEP 1: CONFIGURE THIS SECTION >>>
    #
    # Define the path to the directory containing your .mat files
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!! PLEASE UPDATE THIS PATH to point to your data directory          !!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    data_directory = '/gpfs/data/karczmar-lab/workspaces/rachelgordon/DRO-BreastDCEMRI/data'  # Example: 'C:/Users/YourUser/Documents/DRO_data'
    
    # Define the range of subject files to load
    num_subjects = 53
    #
    # <<< END OF CONFIGURATION SECTION >>>
    #
    
    # --- STEP 2: Load the real data from .mat files ---
    patient_data_list = []
    print(f"Attempting to load data from: {os.path.abspath(data_directory)}")

    for i in range(1, num_subjects + 1):
        filename = f'sub{i}.mat'
        full_path = os.path.join(data_directory, filename)
        
        try:
            # Load the .mat file
            mat_contents = loadmat(full_path)
            
            # Assemble the data into the required dictionary format.
            # The keys ('mask', 'AIF', etc.) must match what gen_dro expects.
            subject_dict = {
                'ID': f'sub_{i}',
                'mask': mat_contents['mask'],
                'AIF': mat_contents['aif'].flatten(),  # .flatten() ensures AIF is 1D
                'S0': mat_contents['S0'],
                'smap': mat_contents['smap']
            }
            patient_data_list.append(subject_dict)
            print(f"Successfully loaded and processed {filename}")

        except FileNotFoundError:
            print(f"Warning: Could not find file {filename}. Skipping.")
        except KeyError as e:
            print(f"Warning: File {filename} is missing a required key: {e}. Skipping.")

    # Check if any data was loaded successfully
    if not patient_data_list:
        raise ValueError(
            "No data was loaded. Please check that 'data_directory' is correct "
            "and that the .mat files exist and contain the required variables."
        )

    print(f"\nSuccessfully loaded {len(patient_data_list)} patient cases.")
    
    # --- STEP 3: Call the gen_dro function ---
    # The gen_dro function will randomly select one case from the loaded data.
    print("Generating DRO... (this may take a minute)")
    dro_results = gen_dro(patient_data_list, num_frames=48)
    print("DRO generation complete.")
    
    # --- STEP 4: Display some results ---
    # Note: 'S0' in dro_results is the one from the randomly selected case.
    sim_img_final_frame = dro_results['simImg'][:, :, -1]
    par_map_ktrans = dro_results['parMap'][:, :, 3]  # ktrans is the 4th parameter

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"DRO Results for Randomly Selected Case (ID: {dro_results['ID']})")
    
    im0 = axes[0].imshow(dro_results['S0'], cmap='gray')
    axes[0].set_title('Input S0')
    fig.colorbar(im0, ax=axes[0])
    
    # Use quantile for robust color scaling in case of outliers
    vmax_ktrans = np.quantile(par_map_ktrans[par_map_ktrans > 0], 0.99)
    im1 = axes[1].imshow(par_map_ktrans, cmap='viridis', vmax=vmax_ktrans)
    axes[1].set_title('Generated Ktrans Map')
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(np.abs(sim_img_final_frame), cmap='gray')
    axes[2].set_title('Simulated Image (Final Frame)')
    fig.colorbar(im2, ax=axes[2])
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the output figure
    output_filename = "dro_results_real_data.png"
    plt.savefig(output_filename)
    print(f"\nSaved results plot to {output_filename}")
    
    # Optionally display the plot
    # plt.show()