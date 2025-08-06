#!/usr/bin/env python3
"""
Cache Manager - Utilities for managing the image cache

This script provides various utilities for managing the cached images.

Usage:
    python cache_manager.py [command] [options]
    
Commands:
    info       - Show cache information
    clear      - Clear all cached images
    download   - Download images to cache (same as cache_downloader.py)
    cleanup    - Force cache cleanup
    
Examples:
    python cache_manager.py info
    python cache_manager.py clear
    python cache_manager.py download 30
    python cache_manager.py cleanup
"""

import sys
import shutil
from pathlib import Path

# Import cache functions from imagecombinator
from imagecombinator import (
    CACHE_DIR,
    init_cache,
    cleanup_cache_if_needed,
    print_cache_info,
    load_cache_metadata,
    save_cache_metadata,
    get_cache_size
)

from cache_downloader import download_images_to_cache

def clear_cache():
    """Clear all cached images and metadata."""
    if not CACHE_DIR.exists():
        print("Cache directory doesn't exist - nothing to clear.")
        return
    
    cache_size = get_cache_size()
    metadata = load_cache_metadata()
    image_count = len(metadata)
    
    print(f"Current cache: {image_count} images, {cache_size:.1f}MB")
    
    response = input("Are you sure you want to clear ALL cached images? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Cache clear cancelled.")
        return
    
    try:
        # Remove all files in cache directory
        for file_path in CACHE_DIR.iterdir():
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
        
        print(f"Cleared {image_count} images ({cache_size:.1f}MB) from cache.")
        
        # Reinitialize empty cache
        init_cache()
        
    except Exception as e:
        print(f"Error clearing cache: {e}")

def force_cleanup():
    """Force cache cleanup regardless of size."""
    print("Forcing cache cleanup...")
    cleanup_cache_if_needed()
    print("Cache cleanup complete.")
    print_cache_info()

def show_help():
    """Show help information."""
    print(__doc__)

def main():
    """Main function to handle command line arguments."""
    
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "info":
        print_cache_info()
        
    elif command == "clear":
        clear_cache()
        
    elif command == "cleanup":
        force_cleanup()
        
    elif command == "download":
        if len(sys.argv) > 2:
            try:
                count = int(sys.argv[2])
                if count <= 0:
                    print("Error: Number of images must be positive")
                    sys.exit(1)
            except ValueError:
                print("Error: Invalid number format")
                sys.exit(1)
        else:
            count = 25  # Default
        
        print(f"Downloading {count} images...")
        downloaded = download_images_to_cache(count)
        print(f"Downloaded {downloaded} images.")
        print_cache_info()
        
    elif command in ["help", "-h", "--help"]:
        show_help()
        
    else:
        print(f"Unknown command: {command}")
        show_help()
        sys.exit(1)

if __name__ == "__main__":
    main()