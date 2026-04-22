"""
Mask augmentation transforms to avoid perfect-edge bias.
Applies jitter, dilate/erode, elastic warp, and partial occlusions on-the-fly.
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, map_coordinates, gaussian_filter
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
        jitter_prob: float = 0.1,
        dilate_erode_prob: float = 0.15,
        elastic_prob: float = 0.05,
        occlusion_prob: float = 0.05,
        jitter_pixels: int = 1,
        morph_kernel_size: int = 2,
        elastic_alpha: float = 5.0,
        elastic_sigma: float = 2.0,
        occlusion_patches: int = 2,
        occlusion_size: int = 10,
        max_augs_per_sample: int = 2,
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
            max_augs_per_sample: Maximum number of augmentations applied to a mask
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
        self.max_augs_per_sample = max(1, int(max_augs_per_sample))
    
    def __call__(self, mask: np.ndarray) -> np.ndarray:
    # The reason i used the call method is to make it easy to integrate into a data pipeline, e.g.:
    # transforms = MaskAugmentation(...)
    # augmented_mask = transforms(mask)
    
        """
        Apply augmentations based on per-transform probabilities.
        Probabilities come from config and must be respected exactly, e.g.
        elastic_prob=0.0 guarantees no elastic transform is applied.
        
        Args:
            mask: (H, W) array with class IDs
            
        Returns:
            Augmented mask (H, W)
        """
        mask = mask.copy()

        # Select augmentations independently according to configured probabilities.
        selected = []
        if random.random() < self.dilate_erode_prob:
            selected.append(self._apply_morphology)
        if random.random() < self.elastic_prob:
            selected.append(self._apply_elastic_transform)
        if random.random() < self.jitter_prob:
            selected.append(self._apply_boundary_jitter)
        if random.random() < self.occlusion_prob:
            selected.append(self._apply_occlusions)

        if not selected:
            return mask

        # Keep distortions controlled when multiple probabilities fire.
        if len(selected) > self.max_augs_per_sample:
            selected = random.sample(selected, self.max_augs_per_sample)

        for aug_func in selected:
            mask = aug_func(mask)
        
        return mask
    
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply random dilation or erosion per class WITHOUT overwriting other classes.
        This prevents class boundary conflicts.
        """
        unique_classes = np.unique(mask)
        out = mask.copy()
        
        # Store dilated/eroded regions per class to resolve conflicts
        class_changes = {}
        
        for cls in unique_classes:
            if cls == 0:  # Skip unlabeled
                continue
            
            class_mask = (mask == cls)
            
            # Randomly choose dilation or erosion
            if random.random() < 0.5:  # 50% dilate, 50% erode
                # Dilate
                # For each class after dilation version.(BUT FOR A SPESFICIC CLASS VERSION)
                new_mask = binary_dilation(
                    class_mask,
                    structure=np.ones((self.morph_kernel_size, self.morph_kernel_size))
                )
                
                # The expansions shows the pixels that changes after dilation.
                expansion = new_mask & ~class_mask
                class_changes[cls] = ('dilate', expansion)
            else:
                # Erode
                new_mask = binary_erosion(
                    class_mask,
                    structure=np.ones((self.morph_kernel_size, self.morph_kernel_size))
                )
                # Mark regions to remove
                erosion = class_mask & ~new_mask
                class_changes[cls] = ('erode', erosion)
        
        # Apply changes: erosions first (make space), then dilations (fill space)
        # First pass: erosions
        for cls, (op, region) in class_changes.items():
            if op == 'erode':
                out[region] = 0  # Set eroded regions to unlabeled
        
        # Second pass: dilations (only into unlabeled regions)
        for cls, (op, region) in class_changes.items():
            if op == 'dilate':
                # Only expand into regions that are now unlabeled
                valid_expansion = region & (out == 0)
                out[valid_expansion] = cls
        
        return out
    
    def _apply_boundary_jitter(self, mask: np.ndarray) -> np.ndarray:
        """
        Add small random offsets to boundary pixels by SWAPPING (not copying).
        This prevents label duplication and maintains topology.
        """
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
        
        # Only jitter 10% of boundary pixels for subtle effect
        out = mask.copy()
        for y, x in boundary_coords[::10]:
            # Random offset
            dy = random.randint(-self.jitter_pixels, self.jitter_pixels)
            dx = random.randint(-self.jitter_pixels, self.jitter_pixels)
            
            new_y = np.clip(y + dy, 0, h - 1)
            new_x = np.clip(x + dx, 0, w - 1)
            
            # SWAP pixels instead of copy to preserve topology
            tmp = out[new_y, new_x]
            out[new_y, new_x] = out[y, x]
            out[y, x] = tmp
        
        return out
    
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

    # Randomly removes small mask regions.    
    def _apply_occlusions(self, mask: np.ndarray) -> np.ndarray:
        """Add random rectangular occlusions (set to unlabeled)."""
        h, w = mask.shape
        if h <= self.occlusion_size or w <= self.occlusion_size:
            return mask
        out = mask.copy()

        for _ in range(self.occlusion_patches):
            y = random.randint(0, h - self.occlusion_size)
            x = random.randint(0, w - self.occlusion_size)
            patch_h = random.randint(self.occlusion_size // 2, self.occlusion_size)
            patch_w = random.randint(self.occlusion_size // 2, self.occlusion_size)
            out[y:y+patch_h, x:x+patch_w] = 0

        return out


class NoAugmentation:
    """Dummy augmentation that does nothing (for validation)."""
    
    def __call__(self, mask: np.ndarray) -> np.ndarray:
        return mask
    
    # we can easily change the MaskAugmentation to NoAugmentation in the data pipeline for validation/testing, e.g.:
    # transforms = NoAugmentation()
