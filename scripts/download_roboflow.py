#!/usr/bin/env python3
"""
Roboflow Dataset Downloader for RoadSignNet-SAL

This script downloads datasets from Roboflow using their API and organizes
them into the required YOLO format structure for training.

Usage:
    python scripts/download_roboflow.py --api-key YOUR_API_KEY --workspace WORKSPACE --project PROJECT --version VERSION
    
    Or set ROBOFLOW_API_KEY environment variable and configure in config.yaml:
    python scripts/download_roboflow.py
    
Example:
    python scripts/download_roboflow.py --api-key abc123 --workspace my-workspace --project road-signs --version 1
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from roboflow import Roboflow
except ImportError:
    print("Error: roboflow package not installed.")
    print("Install it with: pip install roboflow")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("Error: pyyaml package not installed.")
    print("Install it with: pip install pyyaml")
    sys.exit(1)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_roboflow_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    output_dir: str = "./data",
    format: str = "yolov8"
) -> dict:
    """
    Download a dataset from Roboflow.
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project: Roboflow project name
        version: Dataset version number
        output_dir: Output directory for the dataset
        format: Export format (default: yolov8 for YOLO format)
    
    Returns:
        dict: Dataset information including paths
    """
    print(f"\n{'='*60}")
    print("Roboflow Dataset Downloader")
    print(f"{'='*60}")
    print(f"Workspace: {workspace}")
    print(f"Project: {project}")
    print(f"Version: {version}")
    print(f"Format: {format}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Initialize Roboflow
    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    
    # Access the project
    print(f"Accessing project: {workspace}/{project}")
    rf_project = rf.workspace(workspace).project(project)
    
    # Get the specific version
    print(f"Getting version {version}...")
    dataset = rf_project.version(version)
    
    # Download the dataset
    print(f"\nDownloading dataset in {format} format...")
    print("This may take a few minutes depending on dataset size...\n")
    
    # Download to a temporary location first
    download_result = dataset.download(format, location=output_dir)
    
    print(f"\n✓ Dataset downloaded successfully!")
    print(f"  Location: {download_result.location}")
    
    # Get dataset info
    info = {
        "name": project,
        "version": version,
        "location": download_result.location,
        "format": format
    }
    
    return info


def reorganize_dataset(download_location: str, target_dir: str = "./data"):
    """
    Reorganize downloaded Roboflow dataset into standard structure.
    
    Roboflow typically downloads to:
        project-version/train/images, project-version/train/labels, etc.
    
    We reorganize to:
        data/train/images, data/train/labels, etc.
    """
    download_path = Path(download_location)
    target_path = Path(target_dir)
    
    print(f"\nReorganizing dataset structure...")
    print(f"  From: {download_path}")
    print(f"  To: {target_path}")
    
    # Common Roboflow directory structures
    splits = ['train', 'valid', 'test']
    split_mapping = {'valid': 'val'}  # Roboflow uses 'valid', we want 'val'
    
    for split in splits:
        src_split = download_path / split
        if not src_split.exists():
            continue
            
        # Map 'valid' to 'val'
        target_split_name = split_mapping.get(split, split)
        
        # Handle images
        src_images = src_split / 'images'
        if src_images.exists():
            dst_images = target_path / target_split_name / 'images'
            dst_images.mkdir(parents=True, exist_ok=True)
            
            for img_file in src_images.iterdir():
                if img_file.is_file():
                    shutil.copy2(img_file, dst_images / img_file.name)
            
            print(f"  ✓ Copied {split} images to {dst_images}")
        
        # Handle labels
        src_labels = src_split / 'labels'
        if src_labels.exists():
            dst_labels = target_path / target_split_name / 'labels'
            dst_labels.mkdir(parents=True, exist_ok=True)
            
            for label_file in src_labels.iterdir():
                if label_file.is_file():
                    shutil.copy2(label_file, dst_labels / label_file.name)
            
            print(f"  ✓ Copied {split} labels to {dst_labels}")
    
    # Check for data.yaml and extract class names
    data_yaml = download_path / 'data.yaml'
    if data_yaml.exists():
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        if 'names' in data_config:
            class_names = data_config['names']
            num_classes = len(class_names)
            print(f"\n  Dataset Info:")
            print(f"    - Number of classes: {num_classes}")
            print(f"    - Classes: {class_names[:5]}..." if len(class_names) > 5 else f"    - Classes: {class_names}")
            
            return {
                'num_classes': num_classes,
                'class_names': class_names
            }
    
    return None


def update_config_with_classes(config_path: str, dataset_info: dict):
    """Update config.yaml with detected class information."""
    if dataset_info is None:
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update num_classes
    config['model']['num_classes'] = dataset_info['num_classes']
    config['data']['num_classes'] = dataset_info['num_classes']
    config['data']['class_names'] = dataset_info['class_names']
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n  ✓ Updated {config_path} with {dataset_info['num_classes']} classes")


def count_dataset_files(data_dir: str = "./data"):
    """Count images and labels in the dataset."""
    data_path = Path(data_dir)
    
    print(f"\n{'='*60}")
    print("Dataset Summary")
    print(f"{'='*60}")
    
    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        if not split_path.exists():
            continue
        
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'
        
        num_images = len(list(images_dir.glob('*'))) if images_dir.exists() else 0
        num_labels = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
        
        print(f"  {split:>5}: {num_images:>5} images, {num_labels:>5} labels")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets from Roboflow for RoadSignNet-SAL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download with command line arguments:
  python scripts/download_roboflow.py --api-key YOUR_KEY --workspace my-workspace --project road-signs --version 1

  # Use environment variable for API key:
  set ROBOFLOW_API_KEY=your_key
  python scripts/download_roboflow.py --workspace my-workspace --project road-signs --version 1
  
  # Use config file (set roboflow settings in config.yaml):
  python scripts/download_roboflow.py --config config/config.yaml

Popular Road Sign Datasets on Roboflow:
  - workspace: roboflow-100    project: road-signs    (traffic sign detection)
  - workspace: roboflow-100    project: road-traffic  (traffic objects)
        """
    )
    
    parser.add_argument('--api-key', type=str, default=None,
                        help='Roboflow API key (or set ROBOFLOW_API_KEY env var)')
    parser.add_argument('--workspace', type=str, default=None,
                        help='Roboflow workspace name')
    parser.add_argument('--project', type=str, default=None,
                        help='Roboflow project name')
    parser.add_argument('--version', type=int, default=1,
                        help='Dataset version number (default: 1)')
    parser.add_argument('--format', type=str, default='yolov8',
                        choices=['yolov5', 'yolov7', 'yolov8', 'coco', 'voc'],
                        help='Export format (default: yolov8)')
    parser.add_argument('--output', type=str, default='./data',
                        help='Output directory (default: ./data)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Config file path (default: config/config.yaml)')
    parser.add_argument('--update-config', action='store_true', default=True,
                        help='Update config.yaml with dataset info (default: True)')
    parser.add_argument('--no-reorganize', action='store_true',
                        help='Skip reorganizing dataset structure')
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(project_root)
    
    # Get API key
    api_key = args.api_key or os.environ.get('ROBOFLOW_API_KEY')
    
    # Try to load from config if not provided
    workspace = args.workspace
    project = args.project
    version = args.version
    
    if os.path.exists(args.config):
        config = load_config(args.config)
        roboflow_config = config.get('roboflow', {})
        
        if not api_key:
            api_key = roboflow_config.get('api_key')
        if not workspace:
            workspace = roboflow_config.get('workspace')
        if not project:
            project = roboflow_config.get('project')
        if args.version == 1 and 'version' in roboflow_config:
            version = roboflow_config.get('version', 1)
    
    # Validate required parameters
    if not api_key:
        print("Error: Roboflow API key is required.")
        print("Provide it via --api-key argument or ROBOFLOW_API_KEY environment variable")
        print("Or add it to config.yaml under roboflow.api_key")
        print("\nGet your API key at: https://app.roboflow.com/settings/api")
        sys.exit(1)
    
    if not workspace or not project:
        print("Error: Workspace and project names are required.")
        print("Provide via --workspace and --project arguments")
        print("Or add them to config.yaml under roboflow.workspace and roboflow.project")
        print("\nExample: python scripts/download_roboflow.py --workspace roboflow-100 --project road-signs --version 1")
        sys.exit(1)
    
    try:
        # Download dataset
        download_info = download_roboflow_dataset(
            api_key=api_key,
            workspace=workspace,
            project=project,
            version=version,
            output_dir=args.output,
            format=args.format
        )
        
        # Reorganize if needed
        dataset_info = None
        if not args.no_reorganize:
            dataset_info = reorganize_dataset(
                download_info['location'],
                args.output
            )
        
        # Update config with class information
        if args.update_config and dataset_info and os.path.exists(args.config):
            update_config_with_classes(args.config, dataset_info)
        
        # Show summary
        count_dataset_files(args.output)
        
        print("✓ Dataset ready for training!")
        print("\nNext steps:")
        print("  1. Review the dataset in ./data/")
        print("  2. Check config/config.yaml for correct settings")
        print("  3. Start training: python scripts/train.py")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nCommon issues:")
        print("  - Invalid API key: Get yours at https://app.roboflow.com/settings/api")
        print("  - Wrong workspace/project: Check the URL when viewing your dataset")
        print("  - Network issues: Check your internet connection")
        sys.exit(1)


if __name__ == "__main__":
    main()
