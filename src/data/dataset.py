"""
PyTorch Dataset for ControlNet training with segmentation masks and text prompts.
"""

import os
import csv
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T

# Handle both direct execution and module import
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sensation_root = Path(__file__).parent.parent.parent.parent
    if str(sensation_root) not in sys.path:
        sys.path.insert(0, str(sensation_root))
    from DDPM.src.data.transforms import MaskAugmentation, NoAugmentation
    from DDPM.src import config
else:
    from .transforms import MaskAugmentation, NoAugmentation
    from .. import config

import sys
from pathlib import Path
data_prep_dir = Path(__file__).parent.parent.parent / "data_prep"
if str(data_prep_dir) not in sys.path:
    sys.path.insert(0, str(data_prep_dir))
from class_mapping import remap_mask

# Simplified 8-class color palette (replaces 35-class Cityscapes palette)
PALETTE = torch.tensor([
    [0, 0, 0],        # 0: background (void, sky, buildings)
    [128, 64, 128],   # 1: road
    [244, 35, 232],   # 2: walkable (sidewalk, crosswalk, etc)
    [220, 20, 60],    # 3: pedestrian
    [0, 0, 142],      # 4: vehicle
    [220, 220, 0],    # 5: traffic control
    [190, 153, 153],  # 6: obstacle
    [107, 142, 35],   # 7: environment
], dtype=torch.float32) / 255.0


class ControlNetDataset(Dataset):
    """
    Dataset for ControlNet training with (image, mask, prompt) triplets.
    
    Loads:
    - RGB images (for diffusion target)
    - Segmentation masks (for ControlNet conditioning)
    - Text prompts (from CSV)
    
    Applies:
    - Image normalization to [-1, 1] (SD standard)
    - Mask augmentation (jitter, morph, elastic, occlusions)
    - Resize to target resolution
    """
    
    def __init__(
        self,
        images_dir: Optional[str] = None,
        masks_dir: Optional[str] = None,
        prompts_file: Optional[str] = None,
        resolution: int = 512,
        augment_masks: bool = True,
        mask_aug_params: Optional[Dict] = None,
    ):
        """
        Args:
            images_dir: Directory with RGB images (REQUIRED - use helper functions)
            masks_dir: Directory with segmentation masks (REQUIRED - use helper functions)
            prompts_file: CSV file with prompts (REQUIRED - use helper functions)
            resolution: Target resolution for images/masks
            augment_masks: Whether to apply mask augmentation
            mask_aug_params: Parameters for MaskAugmentation (default: use presets)
            
        Note:
            Use create_train_dataset() or create_val_dataset() instead of calling this directly.
        """
        # All paths are now required since we have separate train/test folders
        if images_dir is None:
            raise ValueError("images_dir is required! Use create_train_dataset() or create_val_dataset()")
        if masks_dir is None:
            raise ValueError("masks_dir is required! Use create_train_dataset() or create_val_dataset()")
        if prompts_file is None:
            raise ValueError("prompts_file is required! Use create_train_dataset() or create_val_dataset()")
        
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.prompts_file = Path(prompts_file)
        self.resolution = resolution
        
        # Validate paths
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        if not self.prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_file}")
        
        # Load prompts CSV
        self.data = self._load_prompts_csv()
        
        # Setup transforms - use resolution from config, not hardcoded 512
        self.image_transform = T.Compose([
            T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(resolution),
            T.ToTensor(), # Converts to [0, 1], also converts to CxHxW
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB to [-1, 1] , VAE requires this normalization for stable training
        ])
        
        self.mask_transform = T.Compose([
            T.Resize(resolution, interpolation=T.InterpolationMode.NEAREST),
            T.CenterCrop(resolution),
        ])
        
        # Mask augmentation
        if augment_masks:
            aug_params = mask_aug_params or {}
            self.mask_augmentation = MaskAugmentation(**aug_params)
        else:
            self.mask_augmentation = NoAugmentation()
        
        print(f"Dataset initialized with {len(self.data)} samples")
        print(f"  Images: {self.images_dir}")
        print(f"  Masks: {self.masks_dir}")
        print(f"  Prompts: {self.prompts_file}")
        print(f"  Resolution: {self.resolution}")
        print(f"  Mask augmentation: {augment_masks}")
    
    def _load_prompts_csv(self) -> list:
        """Load prompts from CSV file."""
        data = []
        
        with open(self.prompts_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'image_name': row['Image_Name'],
                    'mask_name': row['Mask_Name'],
                    'prompt': row['Text_Prompt']
                })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            dict with:
            - image: (3, H, W) float tensor in [-1, 1]
            - mask: (H, W) long tensor with class IDs
            - prompt: text string
            - mask_rgb: (3, H, W) float tensor in [0, 1] for ControlNet
        """
        try:
            item = self.data[idx]
            
            # Load image
            image_path = self.images_dir / item['image_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)
            
            # Load mask
            mask_path = self.masks_dir / item['mask_name']
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            mask = Image.open(mask_path)
            mask = self.mask_transform(mask)
            mask = np.array(mask)
            
            # Remap from 35 classes to 8 simplified classes
            mask = remap_mask(mask)
            
            # Apply mask augmentation (on-the-fly)
            mask = self.mask_augmentation(mask)
            
            # Convert to tensor
            mask = torch.from_numpy(mask).long()
            
            # Create RGB visualization of mask for ControlNet using color palette
            mask_rgb = PALETTE[mask]              # (H, W, 3)
            mask_rgb = mask_rgb.permute(2, 0, 1)  # (3, H, W)
            
            return {
                'image': image,
                'mask': mask,
                'prompt': item['prompt'],
                'mask_rgb': mask_rgb,
                'image_name': item['image_name'],
            }
        except Exception as e:
            print(f"[DATASET ERROR] idx={idx}: {e}")
            raise


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader with proper collate function.
    
    Args:
        dataset: ControlNetDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        DataLoader
    """
    
    def collate_fn(batch):
        """Custom collate to handle prompts (strings)."""
        images = torch.stack([item['image'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        prompts = [item['prompt'] for item in batch]
        mask_rgbs = torch.stack([item['mask_rgb'] for item in batch])
        image_names = [item['image_name'] for item in batch]
        
        return {
            'image': images,
            'mask': masks,
            'prompt': prompts,
            'mask_rgb': mask_rgbs,
            'image_name': image_names,
        }
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


def create_train_dataset(resolution: int = 512, **kwargs) -> ControlNetDataset:
    """
    Create dataset for training with mask augmentation from config.
    Uses data/train/images, data/train/labels, and data/train/prompts/prompts.csv.
    
    Args:
        resolution: Target resolution
        **kwargs: Additional arguments for ControlNetDataset
        
    Returns:
        ControlNetDataset with augmentation (train data)
    """
    # Build mask augmentation parameters from config
    mask_aug_params = {
        'jitter_prob': config.MASK_JITTER_PROB,
        'dilate_erode_prob': config.MASK_DILATE_ERODE_PROB,
        'elastic_prob': config.MASK_ELASTIC_PROB,
        'occlusion_prob': config.MASK_OCCLUSION_PROB,
        'jitter_pixels': config.MASK_JITTER_PIXELS,
        'morph_kernel_size': config.MASK_MORPH_KERNEL_SIZE,
        'elastic_alpha': config.MASK_ELASTIC_ALPHA,
        'elastic_sigma': config.MASK_ELASTIC_SIGMA,
        'occlusion_patches': config.MASK_OCCLUSION_PATCHES,
        'occlusion_size': config.MASK_OCCLUSION_SIZE,
    }
    
    return ControlNetDataset(
        images_dir=config.TRAIN_IMAGES_DIR,
        masks_dir=config.TRAIN_MASKS_DIR,
        prompts_file=config.TRAIN_PROMPTS_FILE,
        resolution=resolution,
        augment_masks=config.USE_MASK_AUGMENTATION,
        mask_aug_params=mask_aug_params,
        **kwargs
    )


def create_val_dataset(resolution: int = 512, **kwargs) -> ControlNetDataset:
    """
    Create dataset for validation without augmentation.
    Uses data/test/images, data/test/labels, and data/test/prompts/prompts.csv.
    
    Args:
        resolution: Target resolution
        **kwargs: Additional arguments for ControlNetDataset
        
    Returns:
        ControlNetDataset without augmentation (test data)
    """
    return ControlNetDataset(
        images_dir=config.VAL_IMAGES_DIR,
        masks_dir=config.VAL_MASKS_DIR,
        prompts_file=config.VAL_PROMPTS_FILE,
        resolution=resolution,
        augment_masks=False,
        **kwargs
    )


if __name__ == "__main__":
    """Test dataset loading."""
    print("Testing ControlNetDataset...")
    print()
    
    # Create train dataset
    dataset = ControlNetDataset(
        images_dir=config.TRAIN_IMAGES_DIR,
        masks_dir=config.TRAIN_MASKS_DIR,
        prompts_file=config.TRAIN_PROMPTS_FILE,
        resolution=512,
        augment_masks=True,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print("\nLoading sample batch...")
    
    # Test single sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Image range: [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Mask classes: {torch.unique(sample['mask']).tolist()}")
    print(f"  Mask RGB shape: {sample['mask_rgb'].shape}")
    print(f"  Prompt: {sample['prompt']}")
    print(f"  Image name: {sample['image_name']}")
    
    # Test dataloader
    print("\n" + "=" * 70)
    print("Testing DataLoader with batch_size=2...")
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)
    
    batch = next(iter(dataloader))
    print(f"\nBatch:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Masks: {batch['mask'].shape}")
    print(f"  Prompts: {len(batch['prompt'])} items")
    print(f"    [0]: {batch['prompt'][0]}")
    print(f"    [1]: {batch['prompt'][1]}")
    
    print("\n✓ Dataset test passed!")
