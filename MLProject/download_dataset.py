#!/usr/bin/env python3
"""
Standalone Dataset Downloader for Diabetes Prediction Project
Downloads the latest preprocessed dataset from GitHub releases
"""

import os
import sys
import urllib.request
import pandas as pd
from datetime import datetime

def download_dataset(url, output_path, force_download=False):
    """Download dataset from GitHub releases"""
    
    # Check if file already exists
    if os.path.exists(output_path) and not force_download:
        print(f"✅ Dataset already exists at: {output_path}")
        
        # Check if file is valid
        try:
            df = pd.read_csv(output_path)
            print(f"📊 Existing dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            
            response = input("Dataset exists. Re-download? (y/N): ").lower()
            if response not in ['y', 'yes']:
                return True
        except Exception as e:
            print(f"⚠️  Existing file appears corrupted: {e}")
            print("Will re-download...")
    
    print(f"📥 Downloading dataset from GitHub releases...")
    print(f"🔗 URL: {url}")
    print(f"📁 Output: {output_path}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                sys.stdout.write(f"\r🔄 Progress: {percent}% ({block_num * block_size}/{total_size} bytes)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, progress_hook)
        print()  # New line after progress
        
        # Validate downloaded file
        if not os.path.exists(output_path):
            raise Exception("Downloaded file not found")
        
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise Exception("Downloaded file is empty")
        
        print(f"✅ Download completed successfully!")
        print(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        # Validate CSV structure
        try:
            df = pd.read_csv(output_path)
            print(f"📋 Dataset structure:")
            print(f"   Rows: {df.shape[0]:,}")
            print(f"   Columns: {df.shape[1]}")
            print(f"   Target column: {'Diabetes_binary' if 'Diabetes_binary' in df.columns else 'NOT FOUND!'}")
            
            if 'Diabetes_binary' in df.columns:
                target_dist = df['Diabetes_binary'].value_counts()
                print(f"   Target distribution: {target_dist.to_dict()}")
            
            # Show first few columns
            print(f"   First 5 columns: {list(df.columns[:5])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error validating CSV: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
            print("🗑️  Removed incomplete download")
        return False

def main():
    """Main function"""
    print("🩺 Diabetes Dataset Downloader")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    dataset_url = "https://github.com/wildanmr/Eksperimen_SML_Wildan-Mufid-Ramadhan/releases/latest/download/diabetes_preprocessed.csv"
    output_path = "diabetes_preprocessed.csv"
    
    # Parse command line arguments
    force_download = "--force" in sys.argv or "-f" in sys.argv
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python download_dataset.py [options]")
        print()
        print("Options:")
        print("  --force, -f    Force re-download even if file exists")
        print("  --help, -h     Show this help message")
        print()
        print(f"Source: {dataset_url}")
        print(f"Output: {output_path}")
        return
    
    # Download dataset
    success = download_dataset(dataset_url, output_path, force_download)
    
    print()
    if success:
        print("🎉 Dataset download completed successfully!")
        print(f"📁 File location: {os.path.abspath(output_path)}")
        print()
        print("Next steps:")
        print("  1. Verify the dataset content")
        print("  2. Use in your ML training pipeline")
        print("  3. Run: python modelling.py")
    else:
        print("❌ Dataset download failed!")
        print("Please check your internet connection and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()