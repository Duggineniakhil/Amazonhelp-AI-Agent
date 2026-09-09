"""
Download the Customer Support on Twitter dataset from Kaggle.

Usage:
    python scripts/download_data.py

Requires KAGGLE_API_TOKEN environment variable to be set.
"""

import os
import sys
import zipfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, RAW_CSV_PATH, KAGGLE_API_TOKEN


def download_dataset():
    """Download the Twitter customer support dataset from Kaggle."""
    
    if RAW_CSV_PATH.exists():
        print(f"[✓] Dataset already exists at {RAW_CSV_PATH}")
        print(f"    Size: {RAW_CSV_PATH.stat().st_size / 1e6:.1f} MB")
        return
    
    if not KAGGLE_API_TOKEN:
        print("[✗] KAGGLE_API_TOKEN not set!")
        print("    Set it via: export KAGGLE_API_TOKEN=<your-token>")
        print("    Or add it to your .env file.")
        sys.exit(1)
    
    # Set the environment variable for the kaggle library
    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN
    
    print("[→] Downloading dataset from Kaggle...")
    print("    Dataset: thoughtvector/customer-support-on-twitter")
    print(f"    Target:  {RAW_DATA_DIR}")
    
    try:
        # Try using kaggle API
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            "thoughtvector/customer-support-on-twitter",
            path=str(RAW_DATA_DIR),
            unzip=False,
        )
        
        # Unzip the downloaded file
        zip_path = RAW_DATA_DIR / "customer-support-on-twitter.zip"
        if zip_path.exists():
            print("[→] Extracting ZIP file...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(RAW_DATA_DIR)
            zip_path.unlink()  # Remove zip after extraction
        
        # The CSV might be named differently — find it
        csv_files = list(RAW_DATA_DIR.glob("*.csv"))
        if csv_files and not RAW_CSV_PATH.exists():
            # Rename to expected name
            csv_files[0].rename(RAW_CSV_PATH)
        
        print(f"[✓] Dataset downloaded to {RAW_CSV_PATH}")
        print(f"    Size: {RAW_CSV_PATH.stat().st_size / 1e6:.1f} MB")
        
    except ImportError:
        # Fallback: use kagglehub
        print("[→] kaggle package not available, trying kagglehub...")
        try:
            import kagglehub
            path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
            print(f"[→] Downloaded to: {path}")
            # Copy CSV to our expected location
            import shutil
            downloaded_csvs = list(Path(path).glob("*.csv"))
            if downloaded_csvs:
                shutil.copy2(str(downloaded_csvs[0]), str(RAW_CSV_PATH))
                print(f"[✓] Copied to {RAW_CSV_PATH}")
        except Exception as e:
            print(f"[✗] kagglehub also failed: {e}")
            print("\n[Manual Download Instructions]")
            print("1. Go to: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
            print("2. Click 'Download' to get the ZIP file")
            print(f"3. Extract 'twcs.csv' to: {RAW_DATA_DIR}")
            sys.exit(1)
    
    except Exception as e:
        print(f"[✗] Download failed: {e}")
        print("\n[Manual Download Instructions]")
        print("1. Go to: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
        print("2. Click 'Download' to get the ZIP file")
        print(f"3. Extract 'twcs.csv' to: {RAW_DATA_DIR}")
        sys.exit(1)


if __name__ == "__main__":
    download_dataset()
