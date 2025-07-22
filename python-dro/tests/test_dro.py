from dro import gen_dro
import matplotlib.pyplot as plt
import math 
import numpy as np

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Running Digital Reference Object (DRO) Generation Example ---")
    
    # 1. Create dummy input data for one "patient case"
    img_size = (128, 128) # Use a smaller size for faster testing
    
    # Create S0 (baseline image) and smap (sensitivity map)
    y, x = np.ogrid[:img_size[0], :img_size[1]]
    S0 = np.exp(-((x - img_size[1]/2)**2 + (y - img_size[0]/2)**2) / (2 * (img_size[0]/3)**2)) * 200
    S0 += np.random.rand(*img_size) * 10
    smap = np.ones(img_size)
    
    # Create a dummy integer mask
    # 1=gland, 3=malig, 6=liver, 7=heart
    mask_int = np.zeros(img_size, dtype=int)
    mask_int[30:60, 20:50] = 7  # Heart
    mask_int[30:60, 70:110] = 6 # Liver
    mask_int[70:100, 40:80] = 1 # Glandular
    mask_int[80:95, 85:105] = 3 # Malignant
    
    # Create a dummy AIF
    t_aif = np.linspace(0, 150, 22)
    aif_vals = (np.exp(-(t_aif-30)**2 / 50) * 0.4) + (np.exp(-(t_aif-80)**2 / 800) * 0.2)
    aif_vals[t_aif < 15] = 0

    # Assemble the data into the required list-of-dictionaries format
    patient_data_list = [
        {
            'ID': 'patient_001_dummy',
            'mask': mask_int,
            'AIF': aif_vals,
            'S0': S0,
            'smap': smap
        }
    ]
    
    # 2. Call the gen_dro function
    print("Generating DRO... (this may take a minute)")
    dro_results = gen_dro(patient_data_list)
    print("DRO generation complete.")
    
    # 3. Display some results
    sim_img_final_frame = dro_results['simImg'][:, :, -1]
    par_map_ktrans = dro_results['parMap'][:, :, 3] # ktrans is the 4th parameter
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    im0 = axes[0].imshow(dro_results['S0'], cmap='gray')
    axes[0].set_title('Input S0')
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(par_map_ktrans, cmap='viridis', vmax=np.quantile(par_map_ktrans, 0.99))
    axes[1].set_title('Generated Ktrans Map')
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(np.abs(sim_img_final_frame), cmap='gray')
    axes[2].set_title('Simulated Image (Final Frame)')
    fig.colorbar(im2, ax=axes[2])
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("dro.png")