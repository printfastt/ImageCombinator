#!/usr/bin/env python3
"""
Cache Downloader - Pre-populate the image cache with Flickr images

This script downloads a specified number of images from Flickr and stores them
in the cache for faster access by the image combinator.

Image Selection Criteria:
- Configured in config.py (FLICKR_SEARCH_PARAMS)
- Current: messy, random, blurry, unedited, party, dog, kitchen tags
- Creative Commons licensed, photos only, chronologically sorted
- Extended metadata included (owner, date taken, tags, original URL)

Usage:
    python cache_downloader.py [number_of_images]
    
Example:
    python cache_downloader.py 50
"""

import sys
import random
import requests
from PIL import Image
from io import BytesIO
from config import API_KEY, API_SECRET, URL, FLICKR_SEARCH_PARAMS

# Import cache functions from imagecombinator
from imagecombinator import (
    init_cache, 
    save_image_to_cache, 
    is_image_cached, 
    cleanup_cache_if_needed,
    get_cache_size,
    load_cache_metadata,
    print_cache_info
)

def download_images_to_cache(target_count: int, start_page: int = 1):
    """
    Download specified number of images from Flickr API to cache.
    
    Args:
        target_count: Number of images to download
        start_page: Starting page number for API requests
    """
    print(f"Starting download of {target_count} images to cache...")
    
    # Initialize cache
    init_cache()
    cleanup_cache_if_needed()
    
    # Get current cache state
    metadata = load_cache_metadata()
    existing_photo_ids = set(metadata.keys())
    
    downloaded_count = 0
    current_page = start_page
    max_pages = 50  # Prevent infinite loops
    
    while downloaded_count < target_count and current_page <= max_pages:
        print(f"\nFetching from page {current_page}...")
        
        # Use common parameters from config, with dynamic values
        params = FLICKR_SEARCH_PARAMS.copy()
        params['api_key'] = API_KEY
        params['page'] = current_page
        
        try:
            # Request photos from Flickr
            response = requests.get(URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'photos' not in data or 'photo' not in data['photos']:
                print(f"No photos found on page {current_page}")
                current_page += 1
                continue
            
            photos = data['photos']['photo']
            total_pages = data['photos']['pages']
            
            print(f"Found {len(photos)} photos on page {current_page} of {total_pages}")
            
            # Process each photo
            for photo in photos:
                if downloaded_count >= target_count:
                    break
                    
                photo_id = photo['id']
                
                # Skip if already cached
                if photo_id in existing_photo_ids or is_image_cached(photo_id):
                    continue
                
                try:
                    # Construct Flickr image URL
                    photo_url = f"https://live.staticflickr.com/{photo['server']}/{photo_id}_{photo['secret']}_c.jpg"
                    
                    # Download image
                    print(f"Downloading image {downloaded_count + 1}/{target_count}: {photo_id}")
                    img_response = requests.get(photo_url, timeout=30)
                    img_response.raise_for_status()
                    
                    # Convert to PIL Image
                    img = Image.open(BytesIO(img_response.content)).convert("RGB")
                    
                    # Cache the image
                    save_image_to_cache(img, photo_id, photo)
                    existing_photo_ids.add(photo_id)
                    downloaded_count += 1
                    
                    # Progress update every 10 images
                    if downloaded_count % 10 == 0:
                        cache_size = get_cache_size()
                        print(f"Progress: {downloaded_count}/{target_count} images downloaded, cache size: {cache_size:.1f}MB")
                    
                except requests.exceptions.RequestException as e:
                    print(f"Network error downloading {photo_id}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing image {photo_id}: {e}")
                    continue
            
            current_page += 1
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {current_page}: {e}")
            current_page += 1
            continue
        except Exception as e:
            print(f"Unexpected error on page {current_page}: {e}")
            break
    
    print(f"\nDownload complete! Downloaded {downloaded_count} images.")
    return downloaded_count

def main():
    """Main function to handle command line arguments and run the downloader."""
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            target_count = int(sys.argv[1])
            if target_count <= 0:
                print("Error: Number of images must be positive")
                sys.exit(1)
        except ValueError:
            print("Error: Invalid number format")
            print("Usage: python cache_downloader.py [number_of_images]")
            sys.exit(1)
    else:
        # Default to 25 images if no argument provided
        target_count = 25
        print(f"No argument provided, defaulting to {target_count} images")
    
    # Show initial cache info
    print_cache_info()
    
    print(f"Starting download of {target_count} images...")
    
    try:
        # Start downloading
        downloaded = download_images_to_cache(target_count)
        
        # Show final cache info
        print("\n" + "="*50)
        print("DOWNLOAD SUMMARY")
        print("="*50)
        print(f"Successfully downloaded: {downloaded} images")
        print_cache_info()
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        print_cache_info()
    except Exception as e:
        print(f"\nError during download: {e}")
        sys.exit(1)

def quick_download(count: int):
    """Quick download function for easy scripting."""
    return download_images_to_cache(count)

if __name__ == "__main__":
    main()