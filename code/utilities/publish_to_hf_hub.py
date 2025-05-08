#!/usr/bin/env python3
"""
Script to upload a dataset folder to an existing (private) Hugging Face dataset repo.

Instructions:
1. Install requirements: pip install huggingface_hub
2. Login to the Hugging Face CLI: huggingface-cli login
3. Run: python publish_to_hf_hub.py --folder kk/clean --repo_name reasoning-reciting-probing

Arguments:
  --folder       Path within the results directory to upload (e.g., kk/clean)
  --repo_name    Last part of the repository name (e.g., reasoning-reciting-probing)
  --username     Optional: HF username (default: from environment variable)
"""
import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFolder

def main():
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument("--folder", required=True, help="Path within results directory to upload")
    parser.add_argument("--repo_name", required=True, help="Last part of the repository name")
    parser.add_argument("--username", help="Hugging Face username (default: from environment)")
    
    args = parser.parse_args()
    
    load_dotenv()
    
    # Get username from args or environment
    username = args.username or os.getenv("HF_USERNAME") or "JorisHolshuijsen"
    
    # Construct repo_id and folder_path
    repo_id = f"{username}/{args.repo_name}"
    
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    folder_path = os.path.join(project_root, "results", args.folder)
    
    # Ensure the folder exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")
    
    api = HfApi()
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise RuntimeError("No Hugging Face token found. Run 'huggingface-cli login' first.")
    
    print(f"Uploading {folder_path} to {repo_id} (private repo assumed)...")
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    print("Upload complete!")

if __name__ == "__main__":
    main()