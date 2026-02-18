"""
Utility functions for path management and environment detection.
Makes DDPM package portable across local and HPC environments.
"""

import os
from pathlib import Path
from typing import Optional, Dict


def get_project_root() -> Path:
    """
    Get the DDPM project root directory.
    Works regardless of where the script is called from.
    
    Returns:
        Path: Absolute path to DDPM root directory
    """
    # This file is in DDPM/src/utils.py, so go up 2 levels
    return Path(__file__).parent.parent.absolute()


def detect_environment() -> str:
    """
    Detect if we're running on HPC or local machine.
    
    Returns:
        str: 'hpc' or 'local'
    """
    # Check for common HPC environment variables
    hpc_indicators = [
        'SLURM_JOB_ID',      # SLURM scheduler
        'PBS_JOBID',          # PBS scheduler
        'LSB_JOBID',          # LSF scheduler
        'WORK',               # Common HPC work directory
        'SCRATCH',            # Common HPC scratch directory
    ]
    
    for indicator in hpc_indicators:
        if os.environ.get(indicator):
            return 'hpc'
    
    return 'local'


def get_data_root(env_var: str = 'DDPM_DATA_ROOT', default: Optional[Path] = None) -> Path:
    """
    Get data directory root with fallback logic.
    
    Priority:
    1. Environment variable (e.g., DDPM_DATA_ROOT)
    2. Provided default path
    3. PROJECT_ROOT/data
    
    Args:
        env_var: Environment variable name to check
        default: Default path if env var not set
        
    Returns:
        Path: Absolute path to data directory
    """
    # Check environment variable
    if env_var in os.environ:
        data_path = Path(os.environ[env_var])
        if not data_path.is_absolute():
            # Make it absolute relative to current working directory
            data_path = Path.cwd() / data_path
        return data_path.resolve()
    
    # Use provided default
    if default is not None:
        if not default.is_absolute():
            default = get_project_root() / default
        return default.resolve()
    
    # Final fallback: PROJECT_ROOT/data
    return get_project_root() / "data"


def get_output_root(env_var: str = 'DDPM_OUTPUT_ROOT', default: Optional[Path] = None) -> Path:
    """
    Get output directory root with fallback logic.
    
    Priority:
    1. Environment variable (e.g., DDPM_OUTPUT_ROOT)
    2. Provided default path
    3. PROJECT_ROOT/outputs (inside DDPM folder)
    
    Args:
        env_var: Environment variable name to check
        default: Default path if env var not set
        
    Returns:
        Path: Absolute path to output directory
    """
    # Always use PROJECT_ROOT/outputs unless explicitly overridden
    if env_var in os.environ:
        output_path = Path(os.environ[env_var])
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        return output_path.resolve()
    
    # Use provided default
    if default is not None:
        if not default.is_absolute():
            default = get_project_root() / default
        return default.resolve()
    
    # Final fallback: PROJECT_ROOT/outputs
    return get_project_root() / "outputs"


def get_cache_dir(env_var: str = 'DDPM_CACHE_DIR') -> Path:
    """
    Get cache directory for model weights, etc.
    
    Priority:
    1. Environment variable
    2. $HF_HOME (HuggingFace standard)
    3. ~/.cache/huggingface
    
    Args:
        env_var: Environment variable name to check
        
    Returns:
        Path: Absolute path to cache directory
    """
    # Check custom env var
    if env_var in os.environ:
        cache_path = Path(os.environ[env_var])
        if not cache_path.is_absolute():
            cache_path = Path.cwd() / cache_path
        return cache_path.resolve()
    
    # Check HuggingFace standard
    if 'HF_HOME' in os.environ:
        return Path(os.environ['HF_HOME']).resolve()
    
    # Default to user cache
    return Path.home() / '.cache' / 'huggingface'


def setup_paths_from_env() -> Dict[str, Path]:
    """
    Setup all paths from environment variables or defaults.
    Call this at the start of your script for easy configuration.
    
    Returns:
        dict: Dictionary of all configured paths
    """
    paths = {
        'project_root': get_project_root(),
        'data_root': get_data_root(),
        'output_root': get_output_root(),
        'cache_dir': get_cache_dir(),
        'environment': detect_environment(),
    }
    
    # Derive specific paths
    paths['masks_dir'] = paths['data_root'] / 'labels'
    paths['images_dir'] = paths['data_root'] / 'images'
    paths['prompts_dir'] = paths['data_root'] / 'prompts'
    paths['checkpoint_dir'] = paths['output_root'] / 'checkpoints'
    paths['log_dir'] = paths['output_root'] / 'logs'  # Now inside outputs/ folder
    
    return paths


def create_directory_structure(paths: Dict[str, Path], dirs_to_create: Optional[list] = None):
    """
    Create necessary directories if they don't exist.
    
    Args:
        paths: Dictionary of paths from setup_paths_from_env()
        dirs_to_create: List of directory keys to create (default: all output dirs)
    """
    if dirs_to_create is None:
        dirs_to_create = ['output_root', 'checkpoint_dir', 'log_dir']
    
    for dir_key in dirs_to_create:
        if dir_key in paths:
            paths[dir_key].mkdir(parents=True, exist_ok=True)


def print_environment_info(paths: Optional[Dict[str, Path]] = None):
    """
    Print environment and path information for debugging.
    
    Args:
        paths: Dictionary of paths (if None, will call setup_paths_from_env())
    """
    if paths is None:
        paths = setup_paths_from_env()
    
    print("=" * 80)
    print("DDPM Environment Configuration")
    print("=" * 80)
    print(f"Environment: {paths['environment'].upper()}")
    print(f"Project Root: {paths['project_root']}")
    print(f"Data Root: {paths['data_root']}")
    print(f"Output Root: {paths['output_root']}")
    print(f"Cache Dir: {paths['cache_dir']}")
    print()
    print("Data Directories:")
    print(f"  Images: {paths['images_dir']}")
    print(f"  Masks: {paths['masks_dir']}")
    print(f"  Prompts: {paths['prompts_dir']}")
    print()
    print("Output Directories:")
    print(f"  Checkpoints: {paths['checkpoint_dir']}")
    print(f"  Logs: {paths['log_dir']}")
    print("=" * 80)
    print()


# Convenience function for quick setup
def quick_setup(create_dirs: bool = True, verbose: bool = False) -> Dict[str, Path]:
    """
    Quick setup: detect environment, configure paths, create directories.
    
    Args:
        create_dirs: Whether to create output directories
        verbose: Whether to print environment info
        
    Returns:
        dict: Dictionary of all configured paths
    """
    paths = setup_paths_from_env()
    
    if create_dirs:
        create_directory_structure(paths)
    
    if verbose:
        print_environment_info(paths)
    
    return paths


if __name__ == "__main__":
    # Test the utilities
    print("Testing DDPM path utilities...")
    print()
    
    paths = quick_setup(create_dirs=False, verbose=True)
    
    print("\nChecking data directories:")
    for key in ['images_dir', 'masks_dir', 'prompts_dir']:
        exists = paths[key].exists()
        status = "✓ EXISTS" if exists else "✗ NOT FOUND"
        print(f"  {key}: {status}")
    
    print("\n" + "=" * 80)
    print("Environment Variables You Can Set:")
    print("=" * 80)
    print("DDPM_DATA_ROOT      - Override data directory location")
    print("DDPM_OUTPUT_ROOT    - Override output directory location")
    print("DDPM_CACHE_DIR      - Override model cache directory")
    print("HF_HOME             - HuggingFace cache directory (standard)")
    print()
    print("Example (bash):")
    print("  export DDPM_DATA_ROOT=/path/to/your/data")
    print("  export DDPM_OUTPUT_ROOT=$WORK/outputs")
    print("=" * 80)
