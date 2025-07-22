import numpy as np
import random
import math
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.morphology import disk
import matplotlib.pyplot as plt

def gen_dro(data: list, option: dict = None):
    """
    Generates a Digital Reference Object (DRO) for Dynamic Contrast-Enhanced MRI simulation.

    This is a Python conversion of the MATLAB gen_DRO function.

    Args:
        data (list): A list of dictionaries, where each dictionary represents a patient case
                     and must contain 'mask', 'AIF', 'S0', and 'smap' keys.
        option (dict, optional): A dictionary for optional overrides. Can contain:
                                 'AIF': An Arterial Input Function to use instead of the one from data.
                                 'B1': A B1 field map. Defaults to a uniform map of ones.
                                 'parMap': A pre-computed pharmacokinetic parameter map.

    Returns:
        dict: A dictionary containing all the generated simulation outputs, including:
              'simImg', 'mask', 'parMap', 'smap', 'S0', 'aif', 'T10'.
    """
    if option is None:
        option = {}

    # --- 1. Initialization and Data Unpacking ---
    # Select a random case from the input data list
    idx = random.randint(0, len(data) - 1)
    case_data = data[idx]

    # Constants
    par_var = 0.1
    par_var_t = 0.2
    SIMULATION_DURATION_S = 150.0

    # Unpack data from the selected case
    mask_ = case_data['mask']
    S0 = case_data['S0']
    smap = case_data['smap']#.astype(np.float64)

    aif = option.get('AIF', case_data['AIF'])
    B1 = option.get('B1', np.ones_like(S0))
    # 'ID' is not used in the simulation logic, but we can pass it through
    ID = case_data.get('ID', 'N/A')

    # --- DYNAMICALLY CREATE THE TIME VECTOR ---
    # The number of frames is determined by the length of the loaded AIF.
    num_frames = len(aif)
    # Now, create a time vector that correctly matches the AIF data.
    t = np.linspace(0, SIMULATION_DURATION_S, num_frames)

    if 'parMap' in option:
        parMap = option['parMap']
    else:
        parMap = None # Will be generated later

    nx, ny = S0.shape

    # --- 2. Create Boolean Masks for Tissues ---
    # Tissue mapping: 1-gland, 2-benign, 3-malig, 4-muscle, 5-skin, 6-liver, 7-heart, 8-vasc
    mask = {
        'glandular': (mask_ == 1),
        'benign': (mask_ == 2),
        'malignant': (mask_ == 3),
        'muscle': (mask_ == 4),
        'skin': (mask_ == 5),
        'liver': (mask_ == 6),
        'heart': (mask_ == 7),
        'vascular': (mask_ == 8),
    }

    # Find the number of baseline time points
    nbase_indices = np.where((aif < 0.15) & (t < 300))[0]
    nbase = nbase_indices[-1] + 1 if len(nbase_indices) > 0 else 0

    # Interpolate AIF to a 1-second resolution
    t_1s = np.arange(0, t[-1] + 1)
    # interp_func = interp1d(t, aif, kind='pchip', bounds_error=False, fill_value="extrapolate")
    interp_func = PchipInterpolator(t, aif, extrapolate=True)
    aifci = interp_func(t_1s)

    # --- 3. Generate T10 (Baseline T1) Map ---
    mask_inner = mask['heart'] | mask['liver']
    # Equivalent to MATLAB's strel('disk', 4) and imdilate
    dilated_mask_inner = binary_dilation(mask_inner, structure=disk(4))

    T1_values = {
        'glandular': 1.324, 'malignant': 1.5, 'benign': 1.4,
        'liver': 0.81, 'heart': 0.81, 'muscle': 1.41,
        'skin': 0.85, 'vascular': 1.93 # Original used T1.heart for vascular
    }

    T10 = np.zeros_like(S0, dtype=np.float64)
    for name, t1_val in T1_values.items():
        T10[mask[name]] = t1_val
    T10[T10 == 0] = 1.0  # Set background T1

    # Smooth the T10 map in the inner region
    temp = gaussian_filter(T10, sigma=10)
    T10[dilated_mask_inner] = temp[dilated_mask_inner]

    # --- 4. Generate Pharmacokinetic Parameter Map (if not provided) ---
    if parMap is None:
        p0 = {
            'glandular': np.array([[0,0], [0.010,0.077], [0.1/60,0.115/60], [0.01/60,0.043/60]]),
            'malignant': np.array([[0.101,0.3], [0.131,0.256], [0.259/60,1.032/60], [0.0434/60,1.98/60]]),
            'benign':    np.array([[0.141,0.3], [0.011,0.190], [0.116/60,0.228/60], [0.05/60,1.056/60]]),
            'muscle':    np.array([[0,0], [0.010,0.101], [0.1/60,0.118/60], [0.011/60,0.069/60]]),
            'skin':      np.array([[0,0], [0.039,0.125], [0.1/60,0.151/60], [0.01/60,0.019/60]]),
            'liver':     np.array([[0.1,0.5], [0.353,0.5], [0.433/60,1.227/60], [1.990/60,2/60]]),
            'heart':     np.array([[0.148,0.300], [0.214,0.373], [2/60,2/60], [0.404/60,1.224/60]]),
            'vascular':  np.array([[0,0], [0.3,0.3], [2/60,2/60], [0/60,0/60]])
        }
        ktrans_ranges = {
            'glandular': np.array([0.01,0.0352])/60, 'malignant': np.array([0.0412,0.385])/60,
            'benign':    np.array([0.0453,0.143])/60, 'muscle':    np.array([0.011,0.05])/60,
            'skin':      np.array([0.009,0.017])/60, 'liver':     np.array([0.412,0.979])/60,
            'heart':     np.array([0.365,0.810])/60
        }
        
        # Function to calculate Ktrans from Extended Tofts parameters
        fun_ktrans = lambda x: x[2] * (1 - np.exp(-x[3] / (x[2] + 1e-9))) # Added epsilon for stability

        # Generate base parameters for each tissue type using rejection sampling
        p0_base = {}
        for name, k_range in ktrans_ranges.items():
            par_range = p0[name]
            p_candidate = par_range[:, 0] + (par_range[:, 1] - par_range[:, 0]) * np.random.rand(4)
            ktrans_val = fun_ktrans(p_candidate)
            while not (k_range[0] <= ktrans_val <= k_range[1]):
                p_candidate = par_range[:, 0] + (par_range[:, 1] - par_range[:, 0]) * np.random.rand(4)
                ktrans_val = fun_ktrans(p_candidate)
            p0_base[name] = p_candidate
        
        # Vascular tissue has no ktrans constraint
        p0_base['vascular'] = p0['vascular'][:, 0] + (p0['vascular'][:, 1] - p0['vascular'][:, 0]) * np.random.rand(4)

        # Create spatial parameter maps with noise
        parMap = np.zeros((nx, ny, 4))
        for i in range(4):
            temp_map = np.zeros((nx, ny))
            for name, p_base in p0_base.items():
                # Malignant/benign have higher variance
                variance = par_var_t if name in ['malignant', 'benign'] else par_var
                noise = (1 - variance) + (variance * 2) * np.random.rand(nx, ny)
                rand_map = p_base[i] * noise
                temp_map[mask[name]] = rand_map[mask[name]]
            parMap[:, :, i] = temp_map

    # --- 5. Generate Time-Delayed and Interpolated AIF Maps ---
    aifci_1s = np.zeros((nx, ny, len(t_1s)))
    delays = {'liver': 10, 'glandular': 15, 'vascular': 0, 'heart': 0, 
              'malignant': 3, 'benign': 8, 'muscle': 7, 'skin': 5}
    
    # This loop is slow, but a direct translation. Vectorization is complex.
    for name, delay in delays.items():
        if np.any(mask[name]):
            row_idx, col_idx = np.where(mask[name])
            delayed_aif = np.roll(aifci, delay)
            for r, c in zip(row_idx, col_idx):
                aifci_1s[r, c, :] = delayed_aif
    
    # Interpolate to a finer time resolution for ODE solving
    # ti = np.arange(0, t[-1], 0.1)
    ti = np.arange(0, t[-1] + 0.05, 0.1)
    aifci_Map = np.zeros((nx, ny, len(ti)))
    active_pixels = np.where(parMap[:, :, 1] > 0)
    for r, c in zip(active_pixels[0], active_pixels[1]):
        # interp_func = interp1d(t_1s, aifci_1s[r, c, :], kind='pchip', bounds_error=False, fill_value="extrapolate")
        interp_func = PchipInterpolator(t_1s, aifci_1s[r, c, :], extrapolate=True)
        aifci_Map[r, c, :] = interp_func(ti)

    # --- 6. Spatially Smooth Parameter Maps ---
    parMap[parMap == 0] = 1e-8 # Avoid division by zero
    for i in range(4):
        par_temp = gaussian_filter(parMap[:, :, i], sigma=1)
        temp = parMap[:, :, i].copy()
        temp[~dilated_mask_inner] = 0
        temp = gaussian_filter(temp, sigma=20)
        par_temp[dilated_mask_inner] = temp[dilated_mask_inner]
        parMap[:, :, i] = par_temp

    # --- 7. Solve Kinetic Models ---
    # Downsample fine time points to match original time points
    logIdx = np.zeros_like(ti, dtype=bool)
    start_idx = 0
    for time_pt in t:
        for j in range(start_idx, len(ti)):
            if time_pt <= ti[j]:
                logIdx[j] = True
                start_idx = j
                break

    # Helper for broadcasting
    def add_time_dim(arr): return arr[:, :, np.newaxis]

    # Model 1: Two-Compartment Exchange Model (TCM)
    ve, vp, fp, ktrans = [parMap[:, :, i] for i in range(4)]
    Ce, Cp = [np.zeros_like(aifci_Map) for _ in range(2)]
    for i in range(1, len(ti)):
        dt = ti[i] - ti[i - 1]
        dcp = fp*aifci_Map[:,:,i-1] - (fp+ktrans)*Cp[:,:,i-1] + ktrans*Ce[:,:,i-1]
        dce = ktrans*Cp[:,:,i-1] - ktrans*Ce[:,:,i-1]
        Cp[:,:,i] = Cp[:,:,i-1] + dcp*dt / vp
        Ce[:,:,i] = Ce[:,:,i-1] + dce*dt / ve
    cts_tcm = (Cp * add_time_dim(vp) + Ce * add_time_dim(ve))[:, :, logIdx]

    # Model 2: Extended Patlak Model (EPM)
    ve_epm = 1.0 # In Patlak, ve is effectively infinite
    Ce, Cp = [np.zeros_like(aifci_Map) for _ in range(2)]
    for i in range(1, len(ti)):
        dt = ti[i] - ti[i-1]
        dcp = fp * aifci_Map[:,:,i-1] - (fp + ktrans) * Cp[:,:,i-1]
        dce = ktrans * Cp[:,:,i-1]
        Cp[:,:,i] = Cp[:,:,i-1] + dcp * dt / vp
        Ce[:,:,i] = Ce[:,:,i-1] + dce * dt / ve_epm # This is just `dce * dt`
    # cts_epm = (Cp * add_time_dim(vp) + Ce * add_time_dim(ve_epm))[:, :, logIdx]
    cts_epm = (Cp * add_time_dim(vp) + Ce * ve_epm)[:, :, logIdx]

    # Model 3: Extended Tofts Model
    ve_etofts = 1.0 # In Tofts, Ve is the extravascular space
    Ce = np.zeros_like(aifci_Map)
    for i in range(1, len(ti)):
        dt = ti[i] - ti[i-1]
        dce = ktrans * aifci_Map[:,:,i-1]
        Ce[:,:,i] = Ce[:,:,i-1] + dce * dt / ve_etofts
    cts_eTofts = (aifci_Map * add_time_dim(vp) + Ce)[:, :, logIdx]

    # --- 8. Combine Model Outputs ---
    cts = cts_tcm.copy()
    mask_epm = np.tile(add_time_dim(mask['glandular'] | mask['skin'] | mask['muscle']), (1, 1, cts.shape[2]))
    mask_eTofts = np.tile(add_time_dim(mask['vascular']), (1, 1, cts.shape[2]))
    
    cts[mask_epm] = cts_epm[mask_epm]
    cts[mask_eTofts] = cts_eTofts[mask_eTofts]
    
    nan_mask_cts = np.isnan(cts).sum(axis=2)
    if np.any(nan_mask_cts):
        print(f"Found {np.sum(nan_mask_cts > 0)} NaN voxels in concentration map (cts).")
        cts[np.isnan(cts)] = 0 # Replace NaNs

    # --- 9. Convert Concentration to MRI Signal (SPGR Equation) ---
    TR = 4.87e-3
    theta = 10 * np.pi / 180
    r1 = 4.3
    
    # Apply B1 field inhomogeneity to flip angle
    theta_map = theta * B1

    T1_t = 1.0 / (1.0 / add_time_dim(T10) + r1 * cts)
    E_t = np.exp(-TR / T1_t)
    
    # Use np.newaxis for broadcasting theta_map
    Eh = (1 - E_t) * np.sin(theta_map[:, :, np.newaxis]) / (1 - E_t * np.cos(theta_map[:, :, np.newaxis]))
    
    # Normalize by baseline signal
    baseline_signal = np.mean(Eh[:, :, :nbase], axis=2)
    Eh = Eh / add_time_dim(baseline_signal)
    
    simImg = Eh * add_time_dim(S0)
    
    # --- 10. Debugging Plot for NaN values ---
    nan_mask_sig = np.sum(np.isnan(simImg), axis=2)
    if np.any(nan_mask_sig):
        print(f"Found {np.sum(nan_mask_sig > 0)} NaN voxels in final signal (simImg).")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Location of NaN values in final signal")
        # Plot background image
        ax.imshow(np.abs(S0), cmap='gray')
        # Plot NaN mask on top with transparency
        # The alpha mask makes only the NaN pixels visible
        ax.imshow(nan_mask_sig > 0, cmap='jet', alpha=(nan_mask_sig > 0).astype(float))
        ax.axis('off')
        plt.show()

        # For interactive debugging, like MATLAB's 'keyboard'
        # import pdb; pdb.set_trace()
        
    simImg[np.isnan(simImg)] = 0 # Final cleanup

    # Return all generated data in a dictionary
    return {
        'simImg': simImg, 'mask': mask, 'parMap': parMap, 'smap': smap,
        'S0': S0, 'ID': ID, 'aif': aif, 'T10': T10
    }
