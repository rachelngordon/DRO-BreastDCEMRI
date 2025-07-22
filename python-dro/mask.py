import numpy as np

def fill_rgb_mask(
    image: np.ndarray, 
    binary_mask: np.ndarray, 
    rgb_value: list or tuple
) -> np.ndarray:
    """
    Fills a region of an RGB image with a specified color based on a mask.

    This is the Python/NumPy equivalent of the MATLAB fill_RGBmask function.
    It uses boolean indexing for high efficiency.

    Args:
        image (np.ndarray): The input RGB image (H x W x 3). It will be modified.
        binary_mask (np.ndarray): A 2D boolean or integer array (H x W). 
                                  Non-zero values indicate the region to fill.
        rgb_value (list or tuple): A 3-element list or tuple with the [R, G, B]
                                   color value (e.g., [255, 255, 0]).

    Returns:
        np.ndarray: The modified RGB image.
    """
    # Ensure the mask is boolean for indexing
    boolean_mask = binary_mask.astype(bool)
    
    # Use boolean indexing to set the color in all three channels at once
    image[boolean_mask] = rgb_value
    
    return image


import matplotlib.pyplot as plt

def demonstrate_multiple_colormaps():
    """
    Demonstrates how to display multiple images with different colormaps
    in a single Matplotlib figure. This is the equivalent of what the
    MATLAB `freezeColors` script is used to achieve.
    """
    # Create some sample data
    X = np.random.rand(50, 50)
    Y = np.arange(2500).reshape(50, 50)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    fig.suptitle("Multiple Colormaps in One Figure (Matplotlib)")

    # --- Plot 1 ---
    # Display the first image with the 'hot' colormap
    im1 = ax1.imshow(X, cmap='hot')
    ax1.set_title("Colormap: 'hot'")
    fig.colorbar(im1, ax=ax1, orientation='vertical', shrink=0.8)

    # --- Plot 2 ---
    # Display the second image with the 'viridis' colormap
    # This does NOT affect the first plot.
    im2 = ax2.imshow(Y, cmap='viridis')
    ax2.set_title("Colormap: 'viridis'")
    fig.colorbar(im2, ax=ax2, orientation='vertical', shrink=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Assumes the 'fill_rgb_mask' function from above is defined in the same script
# or imported.

def disp_mask(
    mask_dict: dict, 
    S0: np.ndarray, 
    show_plot: bool = True
) -> tuple:
    """
    Generates a colored mask overlay and displays it on a background image.

    This is a Python conversion of the MATLAB disp_mask function.

    Args:
        mask_dict (dict): A dictionary where keys are tissue names (e.g., 'liver')
                          and values are the corresponding 2D boolean masks.
        S0 (np.ndarray): The 2D grayscale background image.
        show_plot (bool): If True, displays the plots.

    Returns:
        tuple: A tuple containing (S0, temp_overlay) where temp_overlay is the
               generated RGB color mask as a float array (0.0 to 1.0).
    """
    # Initialize a black RGB image of the same size as S0
    # The dtype is uint8 for color values 0-255
    temp_overlay = np.zeros((*S0.shape, 3), dtype=np.uint8)

    # Define colors for each mask
    colors = {
        'liver': [255, 255, 0],
        'heart': [255, 128, 0],
        'glandular': [153, 255, 153],
        'malignant': [255, 0, 0],
        'benign': [255, 0, 0],
        'vascular': [255, 105, 180],
        'skin': [0, 0, 255],
        'muscle': [153, 0, 76]
    }

    # Fill the overlay using the provided masks and colors
    for name, color in colors.items():
        if name in mask_dict and mask_dict[name].any():
            temp_overlay = fill_rgb_mask(temp_overlay, mask_dict[name], color)

    # Convert to float array (0.0-1.0), equivalent to MATLAB's im2double
    temp_float = temp_overlay.astype(np.float64) / 255.0

    if show_plot:
        # Create a figure similar to the MATLAB version
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # --- Subplot 1: Background image with contrast adjustment ---
        bck_img = np.abs(S0)
        # Calculate contrast limits (equivalent to quantile in MATLAB)
        vmax = np.quantile(bck_img.ravel(), 0.98)
        
        ax1.imshow(bck_img, cmap='gray', vmin=0, vmax=vmax)
        ax1.set_title("Background Image (S0) with Contrast")
        ax1.axis('off')

        # --- Subplot 2: Background image with transparent overlay ---
        ax2.imshow(np.abs(S0), cmap='gray')
        
        # Create an alpha channel where the mask is not black
        alpha_channel = (temp_overlay.sum(axis=2) > 0).astype(float)
        
        # Create an RGBA image for the overlay
        # We use the float version for proper alpha blending
        overlay_rgba = np.dstack((
            temp_float[:, :, 0], 
            temp_float[:, :, 1], 
            temp_float[:, :, 2], 
            alpha_channel
        ))
        
        # Display the overlay on top of the background
        ax2.imshow(overlay_rgba)
        ax2.set_title("S0 with Mask Overlay")
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    return S0, temp_float