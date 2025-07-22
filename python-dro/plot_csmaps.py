import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from einops import rearrange # make sure to add this import
from scipy.io import loadmat 

def plot_first_sample_csmaps(subject_id):
    """
    Loads and plots the coil sensitivity maps from the first simulated sample.
    """
    # --- 1. CONFIGURE THE PATH ---
    OUTPUT_DIRECTORY = '/gpfs/data/karczmar-lab/workspaces/rachelgordon/DRO-BreastDCEMRI/data/' #'/ess/scratch/scratch1/rachelgordon/simulated_dataset/'

    print(f"Searching for simulated samples in: {OUTPUT_DIRECTORY}")

    # --- 2. FIND AND LOAD THE DATA FOR THE FIRST SAMPLE ---
    try:
        filename = f'sub{subject_id}.mat'
        full_path = os.path.join(OUTPUT_DIRECTORY, filename)

        # Load the .mat file, which returns a dictionary
        mat_contents = loadmat(full_path)
        
        # Extract the sensitivity maps using the key 'smap'
        # This is based on your data loading example
        smap_data = mat_contents['smap']
        
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        return

    # --- 3. REARRANGE DATA AND EXTRACT MAGNITUDE/PHASE ---
    print("smap shape: ", smap_data.shape)
    
    # rearrange the data from (height, coils, width) to (height, width, coils)
    # this is done on the complex data, before splitting into mag/phase
    # smap_data = rearrange(smap_data, 'h w c -> w h c')
    
    # now that the data is correctly oriented, calculate magnitude and phase
    smap_magnitude = np.abs(smap_data)
    smap_phase = np.angle(smap_data)
    
    # the number of coils is now correctly in the last dimension
    num_coils = smap_data.shape[2]

    # --- 4. PLOT THE MAPS ---
    # your original plotting code from section 4 will now work perfectly
    # with the rearranged data.
    fig, axes = plt.subplots(
        nrows=2, 
        ncols=num_coils, 
        figsize=(3 * num_coils, 6.5)
    )

    fig.suptitle(f"Coil Sensitivity Maps for Subject {subject_id}", fontsize=16)

    for i in range(num_coils):
        # plot magnitude (top row)
        ax_mag = axes[0, i]
        im_mag = ax_mag.imshow(smap_magnitude[:, :, i], cmap='viridis')
        ax_mag.set_title(f'Coil {i+1} Mag')
        ax_mag.axis('off')

        # plot phase (bottom row)
        ax_phase = axes[1, i]
        im_phase = ax_phase.imshow(smap_phase[:, :, i], cmap='twilight')
        ax_phase.set_title(f'Coil {i+1} Phase')
        ax_phase.axis('off')

    plt.savefig('csmaps_rearranged.png')
    print("Saved the corrected plot to 'csmaps_rearranged.png'")


if __name__ == '__main__':
    subject_id = 1
    plot_first_sample_csmaps(subject_id)