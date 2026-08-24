"""
Roboflow Dataset Downloader

Downloads datasets from Roboflow with configurable settings.
Supports both command-line arguments and YAML configuration.
"""
import argparse
import yaml
from pathlib import Path
import os


def load_download_config(config_path=None):
    """
    Load download configuration from YAML file

    Args:
        config_path (str): Path to YAML config file

    Returns:
        dict: Download configuration
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def download_dataset(
    api_key=None,
    workspace=None,
    project=None,
    version=None,
    format='folder',
    location=None,
    config_path=None
):
    """
    Download dataset from Roboflow

    Args:
        api_key (str): Roboflow API key
        workspace (str): Workspace name
        project (str): Project name
        version (int): Dataset version
        format (str): Download format (folder, yolov5, coco, etc.)
        location (str): Download destination path
        config_path (str): Path to YAML config file

    Returns:
        str: Path to downloaded dataset
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow package not installed")
        print("Install it with: pip install roboflow")
        return None

    # Load config from YAML if provided
    config = load_download_config(config_path)

    # Command-line arguments override config file
    api_key = api_key or config.get('api_key') or os.getenv('ROBOFLOW_API_KEY')
    workspace = workspace or config.get('workspace')
    project = project or config.get('project')
    version = version or config.get('version', 1)
    format = format or config.get('format', 'folder')
    location = location or config.get('location', 'data')

    # Validate required parameters
    if not api_key:
        print("[ERROR] API key is required")
        print("Provide via: --api-key, config file, or ROBOFLOW_API_KEY env variable")
        return None

    if not all([workspace, project]):
        print("[ERROR] workspace and project are required")
        return None

    print("=" * 70)
    print("ROBOFLOW DATASET DOWNLOAD")
    print("=" * 70)
    print(f"Workspace: {workspace}")
    print(f"Project: {project}")
    print(f"Version: {version}")
    print(f"Format: {format}")
    print(f"Location: {location}")
    print()

    try:
        # Initialize Roboflow
        print("[INFO] Connecting to Roboflow...")
        rf = Roboflow(api_key=api_key)

        # Get project
        print(f"[INFO] Fetching project '{project}'...")
        proj = rf.workspace(workspace).project(project)

        # Get version
        print(f"[INFO] Downloading version {version}...")
        ver = proj.version(version)

        # Download dataset
        dataset = ver.download(format, location=location)

        print()
        print("=" * 70)
        print("[SUCCESS] DOWNLOAD COMPLETED")
        print("=" * 70)
        print(f"Dataset location: {dataset.location}")
        print()

        return dataset.location

    except Exception as e:
        print(f"[ERROR] Error during download: {str(e)}")
        return None


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description='Download datasets from Roboflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using command-line arguments
  python scripts/download_roboflow.py --api-key YOUR_KEY --workspace cpecgm3 --project rock-paper-scissor-p13xv --version 2

  # Using config file
  python scripts/download_roboflow.py --config configs/download/roboflow.yaml

  # Using environment variable for API key
  export ROBOFLOW_API_KEY=YOUR_KEY
  python scripts/download_roboflow.py --workspace cpecgm3 --project rock-paper-scissor-p13xv --version 2
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Roboflow API key (or use ROBOFLOW_API_KEY env variable)'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        help='Roboflow workspace name'
    )
    parser.add_argument(
        '--project',
        type=str,
        help='Roboflow project name'
    )
    parser.add_argument(
        '--version',
        type=int,
        default=None,
        help='Dataset version (default: from config or 1)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default=None,
        choices=['folder', 'yolov5', 'coco', 'voc', 'tensorflow'],
        help='Download format (default: from config or folder)'
    )
    parser.add_argument(
        '--location',
        type=str,
        default=None,
        help='Download destination path (default: from config or data/)'
    )

    args = parser.parse_args()

    # Download dataset
    result = download_dataset(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        format=args.format,
        location=args.location,
        config_path=args.config
    )

    if result:
        print("[INFO] Next steps:")
        print(f"   1. Update your dataset config to point to: {result}")
        print("   2. Run training: python main.py --config configs/experiments/vit_rps_quick.yaml")
    else:
        exit(1)


if __name__ == "__main__":
    main()
