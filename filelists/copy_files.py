#!/usr/bin/env python3
import argparse
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("xrdcp_transfer.log"), logging.StreamHandler()],
)


def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Copy CMS files using xrdcp")
    parser.add_argument(
        "-i", "--input", required=True, help="Input JSON file with file paths"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory to copy files to"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=4, help="Number of parallel jobs (default: 4)"
    )
    parser.add_argument("-d", "--dataset", help="Only process this specific dataset")
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Dry run, print commands without executing",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite if files already exist",
    )
    return parser.parse_args()


def copy_file(src, dest, force=False, dry_run=False):
    """Copy a single file using xrdcp."""
    # Create output directory structure if it doesn't exist
    # os.makedirs(os.path.dirname(dest), exist_ok=True)

    # # Skip if file already exists and force is not enabled
    # if os.path.exists(dest) and not force:
    #     logging.info(f"File already exists, skipping: {dest}")
    #     return True

    # Construct xrdcp command
    # cmd = ["xrdcp", "--nopbar", "--silent", src, dest]
    cmd = ["xrdcp", src, dest]

    if dry_run:
        logging.info(f"DRY RUN: {' '.join(cmd)}")
        return True

    # Execute command
    logging.info(f"Copying: {src} -> {dest}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"Successfully copied to {dest}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to copy {src}: {e.stderr.strip()}")
        return False


def process_dataset(dataset_name, file_list, output_dir, force=False, dry_run=False):
    """Process all files for a dataset."""
    logging.info(f"Processing dataset: {dataset_name}")
    success_count = 0

    # Create a sanitized directory name from the dataset
    dataset_dir = dataset_name.replace("+", "_").replace("/", "__")
    dataset_output_dir = os.path.join(output_dir, dataset_name)

    for file_path in file_list:
        # Extract filename from the full path
        filename = os.path.basename(file_path)
        dest_path = os.path.join(dataset_output_dir, filename)

        if copy_file(file_path, dest_path, force, dry_run):
            success_count += 1

    logging.info(
        f"Dataset {dataset_name}: {success_count}/{len(file_list)} files copied successfully"
    )
    return success_count


def main():
    args = setup_args()

    # Ensure output directory exists
    if not args.output.startswith("root://"):
        os.makedirs(args.output, exist_ok=True)

    # Load the JSON file with dataset information
    try:
        with open(args.input) as f:
            datasets = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logging.error(f"Error reading input JSON file: {e}")
        return 1

    # Filter dataset if specified
    if args.dataset:
        if args.dataset in datasets:
            datasets = {args.dataset: datasets[args.dataset]}
        else:
            logging.error(f"Dataset {args.dataset} not found in input file")
            return 1

    total_datasets = len(datasets)
    total_files = sum(len(files) for files in datasets.values())
    logging.info(f"Found {total_datasets} datasets with a total of {total_files} files")

    # Process datasets in parallel
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = []
        for dataset_name, file_list in datasets.items():
            futures.append(
                executor.submit(
                    process_dataset,
                    dataset_name,
                    file_list,
                    args.output,
                    args.force,
                    args.dry_run,
                )
            )

        # Wait for all transfers to complete
        success_files = 0
        for future in futures:
            success_files += future.result()

    logging.info(
        f"Transfer complete: {success_files}/{total_files} files copied successfully"
    )
    return 0


if __name__ == "__main__":
    exit(main())
