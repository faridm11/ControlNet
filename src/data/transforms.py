"""
Mask augmentation transforms to avoid perfect-edge bias.
Applies jitter, dilate/erode, elastic warp, and partial occlusions on-the-fly.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, map_coordinates, gaussian_filter
import random


class MaskAugmentation:
    """
    Apply realistic imperfections to segmentation masks during training.
    
    Techniques:
    - Boundary jitter: Small random pixel shifts along edges
    - Morphological ops: Random dilation/erosion to simulate annotation uncertainty
    - Elastic deformation: Subtle warping to break perfect boundaries
    - Partial occlusions: Random small regions set to unlabeled
    """
    
    def __init__(
        self,
        jitter_prob: float = 0.3,
        dilate_erode_prob: float = 0.4,
        elastic_prob: float = 0.2,
        occlusion_prob: float = 0.1,
        jitter_pixels: int = 2,
        morph_kernel_size: int = 3,
        elastic_alpha: float = 10.0,
        elastic_sigma: float = 3.0,
        occlusion_patches: int = 5,
        occlusion_size: int = 20,
    ):
        """
        Args:
            jitter_prob: Probability of applying boundary jitter
            dilate_erode_prob: Probability of morphological operations
            elastic_prob: Probability of elastic deformation
            occlusion_prob: Probability of adding occlusions
            jitter_pixels: Max pixels for boundary jitter
            morph_kernel_size: Kernel size for dilation/erosion
            elastic_alpha: Elastic deformation strength
            elastic_sigma: Elastic deformation smoothness
            occlusion_patches: Number of occlusion patches
            occlusion_size: Size of each occlusion patch
        """
        self.jitter_prob = jitter_prob
        self.dilate_erode_prob = dilate_erode_prob
        self.elastic_prob = elastic_prob
        self.occlusion_prob = occlusion_prob
        self.jitter_pixels = jitter_pixels
        self.morph_kernel_size = morph_kernel_size
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.occlusion_patches = occlusion_patches
        self.occlusion_size = occlusion_size
    
    def __call__(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to mask.
        
        Args:
            mask: (H, W) array with class IDs
            
        Returns:
            Augmented mask (H, W)
        """
        mask = mask.copy()
        
        # 1. Morphological operations (dilate/erode)
        if random.random() < self.dilate_erode_prob:
            mask = self._apply_morphology(mask)
        
        # 2. Elastic deformation
        if random.random() < self.elastic_prob:
            mask = self._apply_elastic_transform(mask)
        
        # 3. Boundary jitter (after elastic to preserve effect)
        if random.random() < self.jitter_prob:
            mask = self._apply_boundary_jitter(mask)
        
        # 4. Random occlusions
        if random.random() < self.occlusion_prob:
            mask = self._apply_occlusions(mask)
        
        return mask
    
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Apply random dilation or erosion per class."""
        unique_classes = np.unique(mask)
        out = mask.copy()
        
        for cls in unique_classes:
            if cls == 0:  # Skip unlabeled
                continue
            
            class_mask = (mask == cls)
            
            # Randomly choose dilation or erosion
            if random.random() < 0.5:
                # Dilate
                new_mask = binary_dilation(
                    class_mask,
                    structure=np.ones((self.morph_kernel_size, self.morph_kernel_size))
                )
            else:
                # Erode
                new_mask = binary_erosion(
                    class_mask,
                    structure=np.ones((self.morph_kernel_size, self.morph_kernel_size))
                )
            
            # Clear old class and write new one
            out[mask == cls] = 0
            out[new_mask] = cls
        
        return out
    
    def _apply_boundary_jitter(self, mask: np.ndarray) -> np.ndarray:
        """Add small random offsets to boundary pixels."""
        # Find boundary pixels (where neighbors have different class)
        h, w = mask.shape
        boundaries = np.zeros_like(mask, dtype=bool)
        
        # Check 4-connected neighbors
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            boundaries |= (mask != shifted)
        
        # Jitter boundary pixels (subsample for performance)
        boundary_coords = list(zip(*np.where(boundaries)))
        random.shuffle(boundary_coords)
        
        # Only jitter 25% of boundary pixels to avoid bottleneck
        for y, x in boundary_coords[::4]:
            # Random offset
            dy = random.randint(-self.jitter_pixels, self.jitter_pixels)
            dx = random.randint(-self.jitter_pixels, self.jitter_pixels)
            
            new_y = np.clip(y + dy, 0, h - 1)
            new_x = np.clip(x + dx, 0, w - 1)
            
            # Swap values
            mask[new_y, new_x] = mask[y, x]
        
        return mask
    
    def _apply_elastic_transform(self, mask: np.ndarray) -> np.ndarray:
        """Apply elastic deformation to mask."""
        h, w = mask.shape
        
        # Generate random displacement fields
        dx = gaussian_filter(
            (np.random.rand(h, w) * 2 - 1),
            self.elastic_sigma,
            mode="constant",
            cval=0
        ) * self.elastic_alpha
        
        dy = gaussian_filter(
            (np.random.rand(h, w) * 2 - 1),
            self.elastic_sigma,
            mode="constant",
            cval=0
        ) * self.elastic_alpha
        
        # Create meshgrid
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Apply displacement
        indices = (
            np.clip(y + dy, 0, h - 1).reshape(-1),
            np.clip(x + dx, 0, w - 1).reshape(-1)
        )
        
        # Remap mask (use nearest neighbor to preserve class IDs)
        mask_warped = map_coordinates(mask, indices, order=0, mode='nearest')
        mask_warped = mask_warped.reshape(h, w)
        
        return mask_warped.astype(mask.dtype)
    
    def _apply_occlusions(self, mask: np.ndarray) -> np.ndarray:
        """Add random rectangular occlusions (set to unlabeled)."""
        h, w = mask.shape
        
        for _ in range(self.occlusion_patches):
            # Random position
            y = random.randint(0, h - self.occlusion_size)
            x = random.randint(0, w - self.occlusion_size)
            
            # Random size (up to max)
            patch_h = random.randint(self.occlusion_size // 2, self.occlusion_size)
            patch_w = random.randint(self.occlusion_size // 2, self.occlusion_size)
            
            # Set to unlabeled (0)
            mask[y:y+patch_h, x:x+patch_w] = 0
        
        return mask


class NoAugmentation:
    """Dummy augmentation that does nothing (for validation)."""
    
    def __call__(self, mask: np.ndarray) -> np.ndarray:
        return mask
