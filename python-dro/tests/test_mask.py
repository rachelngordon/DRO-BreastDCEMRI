import numpy as np
import matplotlib.pyplot as plt
from mask import fill_rgb_mask, disp_mask

# --- Main execution block to test the functions ---
if __name__ == '__main__':
    
    # 1. Create dummy data similar to what the MATLAB script expects
    img_size = (320, 320)
    
    # Create a background image S0 with some gradients
    x, y = np.meshgrid(np.linspace(-1, 1, img_size[1]), np.linspace(-1, 1, img_size[0]))
    S0 = np.exp(-(x**2 + y**2) * 3) * 200 + np.random.rand(*img_size) * 30
    
    # Create a dictionary of dummy boolean masks
    mask_dictionary = {}
    
    # Liver mask (a square)
    mask_dictionary['liver'] = np.zeros(img_size, dtype=bool)
    mask_dictionary['liver'][50:120, 50:150] = True
    
    # Heart mask (a circle)
    cy, cx = 100, 220
    radius = 40
    mask_dictionary['heart'] = (x - (cx-img_size[1]/2)/(img_size[1]/2))**2 + \
                               (y - (cy-img_size[0]/2)/(img_size[0]/2))**2 < (radius/img_size[0])**2

    # Malignant lesion (smaller circle)
    cy, cx = 200, 180
    radius = 25
    mask_dictionary['malignant'] = (x - (cx-img_size[1]/2)/(img_size[1]/2))**2 + \
                                   (y - (cy-img_size[0]/2)/(img_size[0]/2))**2 < (radius/img_size[0])**2
                                   
    # Muscle mask
    mask_dictionary['muscle'] = np.zeros(img_size, dtype=bool)
    mask_dictionary['muscle'][250:280, 20:300] = True
    
    print("Generated dummy data. Calling disp_mask...")
    
    # 2. Call the main function to process and display the data
    returned_S0, returned_overlay = disp_mask(mask_dictionary, S0, show_plot=True)
    
    print("\nScript finished.")
    print(f"Shape of returned S0: {returned_S0.shape}")
    print(f"Shape of returned overlay: {returned_overlay.shape}")