import torch
import torchkbnufft as tkbn
import numpy as np


class MCNUFFT:
    """
    A Python class for Multi-Coil NUFFT, updated with the correct torchkbnufft API.
    
    The smaps are passed to the forward/adjoint calls, not the operator init.
    """
    def __init__(self, k, w, b1, device='cpu'):
        """
        Initializes the MCNUFFT operator.

        Args:
            k (np.ndarray): K-space trajectory. Shape: (points, spokes, frames).
            w (np.ndarray): Density compensation weights. Shape: (points, spokes, frames).
            b1 (np.ndarray): Coil sensitivity maps. Shape: (H, W, C).
            device (str): The device to run on ('cpu' or 'cuda:X').
        """
        self.device = torch.device(device)
        
        im_size = (b1.shape[0], b1.shape[1])
        num_frames = k.shape[2]
        
        # Determine grid size, ensuring standard python integers
        base_grid_size = np.array(im_size) * 1.5 if im_size[0] > 256 else np.array(im_size) * 2.0
        grid_size = tuple(int(dim) for dim in np.floor(base_grid_size))
            
        print(f"Initializing MCNUFFT: Image Size={im_size}, Grid Size={grid_size}, Frames={num_frames}")

        # --- Store smaps and density compensation as tensors ---
        # smaps shape: (C, H, W)
        self.smaps = torch.tensor(b1, dtype=torch.complex64).permute(2, 0, 1).to(self.device)
        # w shape: (points, spokes, frames)
        self.w = torch.tensor(w, dtype=torch.float32).sqrt().to(self.device)

        # --- Create a SINGLE, coil-agnostic forward and adjoint operator ---
        self.nufft_op = tkbn.KbNufft(im_size=im_size, grid_size=grid_size).to(self.device)
        self.adj_nufft_op = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size).to(self.device)
            
        # --- Pre-process and store all trajectories ---
        self.trajs = []
        for t in range(num_frames):
            k_frame = k[:, :, t].flatten()
            # Trajectory shape: (2, M) in range [-pi, pi]
            om = np.stack([np.imag(k_frame), np.real(k_frame)], axis=0) * 2 * np.pi
            self.trajs.append(torch.tensor(om, dtype=torch.float32).to(self.device))
            
    def forward(self, x):
        """
        Performs the forward NUFFT operation (image -> k-space).
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.complex64).to(self.device)
            
        # from (H, W, T) -> (T, H, W)
        x_batched = x.permute(2, 0, 1)
        
        kspace_frames = []
        for t in range(x_batched.shape[0]):
            # Input shape for nufft_op: (batch=1, coils=1, H, W)
            # The coil dimension is 1 because we are passing smaps separately.
            img_t = x_batched[t].unsqueeze(0).unsqueeze(1)
            traj_t = self.trajs[t]
            
            # Pass smaps to the forward call
            # Output kspace_t shape: (batch=1, C, M)
            kspace_t = self.nufft_op(img_t, traj_t, smaps=self.smaps)
            kspace_frames.append(kspace_t)
            
        # Concatenate along the batch dimension (dim=0)
        # Final shape: (T, C, M)
        return torch.cat(kspace_frames, dim=0)

    def adjoint(self, y):
        """
        Performs the adjoint NUFFT operation (k-space -> image).
        """
        # y shape is (T, C, M)
        num_frames = y.shape[0]
        image_frames = []
        
        for t in range(num_frames):
            # k-space for this frame, shape (batch=1, C, M)
            kspace_t = y[t].unsqueeze(0)
            traj_t = self.trajs[t]
            
            # Density comp weights for this frame, shape (M,)
            w_t = self.w[:,:,t].flatten()
            
            # Pass smaps to the adjoint call
            # Adjoint recon, output shape: (batch=1, coils=1, H, W)
            image_t = self.adj_nufft_op(kspace_t * w_t, traj_t, smaps=self.smaps)
            
            # Squeeze the coil dimension before appending
            image_frames.append(image_t.squeeze(1))
            
        # Concatenate along batch dim and permute to (H, W, T)
        return torch.cat(image_frames, dim=0).permute(1, 2, 0)

