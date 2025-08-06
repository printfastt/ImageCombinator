import random
import os
import json
import hashlib
from pathlib import Path
from PIL import Image, ImageDraw
from io import BytesIO
import requests
from config import API_KEY, API_SECRET, URL, FLICKR_SEARCH_PARAMS, IMAGE_SOURCE_MODE

# Cache configuration
CACHE_DIR = Path("image_cache")
CACHE_METADATA_FILE = CACHE_DIR / "metadata.json"
MAX_CACHE_SIZE_MB = 500  # Maximum cache size in MB

def init_cache():
    """Initialize the cache directory and metadata file."""
    CACHE_DIR.mkdir(exist_ok=True)
    if not CACHE_METADATA_FILE.exists():
        with open(CACHE_METADATA_FILE, 'w') as f:
            json.dump({}, f)

def load_cache_metadata() -> dict:
    """Load cache metadata from file."""
    init_cache()
    try:
        with open(CACHE_METADATA_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def save_cache_metadata(metadata: dict):
    """Save cache metadata to file."""
    init_cache()
    with open(CACHE_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def get_image_cache_path(photo_id: str) -> Path:
    """Get the cache file path for a given photo ID."""
    return CACHE_DIR / f"{photo_id}.jpg"

def is_image_cached(photo_id: str) -> bool:
    """Check if an image is already cached."""
    cache_path = get_image_cache_path(photo_id)
    return cache_path.exists()

def save_image_to_cache(image: Image.Image, photo_id: str, photo_info: dict):
    """Save an image to cache with metadata."""
    init_cache()
    cache_path = get_image_cache_path(photo_id)
    
    # Save the image
    image.save(cache_path, "JPEG", quality=85)
    
    # Update metadata
    metadata = load_cache_metadata()
    metadata[photo_id] = {
        'file_path': str(cache_path),
        'file_size': cache_path.stat().st_size,
        'photo_info': photo_info,
        'cached_at': str(Path(cache_path).stat().st_mtime)
    }
    save_cache_metadata(metadata)

def load_image_from_cache(photo_id: str) -> Image.Image:
    """Load an image from cache."""
    cache_path = get_image_cache_path(photo_id)
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")
    raise FileNotFoundError(f"Cached image not found: {photo_id}")

def get_cache_size() -> float:
    """Get the total cache size in MB."""
    if not CACHE_DIR.exists():
        return 0
    
    total_size = 0
    for file_path in CACHE_DIR.rglob("*.jpg"):
        total_size += file_path.stat().st_size
    
    return total_size / (1024 * 1024)  # Convert to MB

def cleanup_cache_if_needed():
    """Clean up cache if it exceeds the maximum size."""
    if get_cache_size() > MAX_CACHE_SIZE_MB:
        print(f"Cache size exceeds {MAX_CACHE_SIZE_MB}MB, cleaning up oldest files...")
        
        metadata = load_cache_metadata()
        
        # Sort by cached_at timestamp
        sorted_items = sorted(
            metadata.items(), 
            key=lambda x: float(x[1].get('cached_at', 0))
        )
        
        # Remove oldest 25% of files
        files_to_remove = len(sorted_items) // 4
        for photo_id, info in sorted_items[:files_to_remove]:
            cache_path = Path(info['file_path'])
            if cache_path.exists():
                cache_path.unlink()
            del metadata[photo_id]
        
        save_cache_metadata(metadata)
        print(f"Cleaned up {files_to_remove} cached images")

def print_cache_info():
    """Print detailed cache information."""
    if not CACHE_DIR.exists():
        print("Cache not initialized")
        return
    
    cache_size = get_cache_size()
    metadata = load_cache_metadata()
    cached_images_count = len(metadata)
    
    print(f"\n--- Cache Information ---")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Cached images: {cached_images_count}")
    print(f"Total cache size: {cache_size:.1f}MB")
    print(f"Maximum cache size: {MAX_CACHE_SIZE_MB}MB")
    print(f"Cache usage: {(cache_size/MAX_CACHE_SIZE_MB)*100:.1f}%")
    
    if cached_images_count > 0:
        print(f"Images cached: {list(metadata.keys())[:5]}{'...' if cached_images_count > 5 else ''}")
    print("--- End Cache Info ---\n")

def generate_random_lines(size: tuple[int, int], num_lines: int = 3) -> list[list[tuple[int, int]]]:
    """
    Generates random jagged/curved paths that cross the canvas.
    Returns a list of paths, each defined by multiple waypoints.
    """
    width, height = size
    lines = []
    
    for _ in range(num_lines):
        # Generate random lines that go from one edge to another
        edge_start = random.randint(0, 3)  # 0=top, 1=right, 2=bottom, 3=left
        edge_end = random.randint(0, 3)
        
        # Ensure start and end are on different edges for better coverage
        while edge_end == edge_start:
            edge_end = random.randint(0, 3)
        
        # Get coordinates for start point
        if edge_start == 0:  # top edge
            start = (random.randint(0, width), 0)
        elif edge_start == 1:  # right edge
            start = (width, random.randint(0, height))
        elif edge_start == 2:  # bottom edge
            start = (random.randint(0, width), height)
        else:  # left edge
            start = (0, random.randint(0, height))
        
        # Get coordinates for end point
        if edge_end == 0:  # top edge
            end = (random.randint(0, width), 0)
        elif edge_end == 1:  # right edge
            end = (width, random.randint(0, height))
        elif edge_end == 2:  # bottom edge
            end = (random.randint(0, width), height)
        else:  # left edge
            end = (0, random.randint(0, height))
        
        # Create a randomized path between start and end
        path = create_randomized_path(start, end, width, height)
        lines.append(path)
    
    return lines

def create_randomized_path(start: tuple[int, int], end: tuple[int, int], width: int, height: int, num_waypoints: int = None) -> list[tuple[int, int]]:
    """
    Creates a randomized path between two points with multiple waypoints.
    """
    if num_waypoints is None:
        num_waypoints = random.randint(2, 5)  # Random number of waypoints
    
    path = [start]
    
    # Calculate the straight-line path as a baseline
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    # Create waypoints along the path with random deviations
    for i in range(1, num_waypoints + 1):
        # Linear interpolation for the base position
        t = i / (num_waypoints + 1)
        base_x = start[0] + t * dx
        base_y = start[1] + t * dy
        
        # Add random deviation perpendicular to the line
        max_deviation = min(width, height) // 6  # Maximum deviation
        deviation = random.randint(-max_deviation, max_deviation)
        
        # Calculate perpendicular direction
        if dx == 0:  # vertical line
            waypoint = (int(base_x + deviation), int(base_y))
        elif dy == 0:  # horizontal line
            waypoint = (int(base_x), int(base_y + deviation))
        else:
            # For diagonal lines, add deviation perpendicular to the line direction
            line_length = (dx**2 + dy**2)**0.5
            perp_x = -dy / line_length
            perp_y = dx / line_length
            
            waypoint = (
                int(base_x + deviation * perp_x),
                int(base_y + deviation * perp_y)
            )
        
        # Ensure waypoint stays within canvas bounds
        waypoint = (
            max(0, min(width, waypoint[0])),
            max(0, min(height, waypoint[1]))
        )
        
        path.append(waypoint)
    
    path.append(end)
    return path

def point_line_side(point: tuple[int, int], line_segment: tuple[tuple[int, int], tuple[int, int]]) -> int:
    """
    Determines which side of a line segment a point is on.
    Returns: 1 for one side, -1 for the other side, 0 for on the line.
    """
    x, y = point
    (x1, y1), (x2, y2) = line_segment
    
    # Calculate cross product to determine side
    cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    
    if cross_product > 0:
        return 1
    elif cross_product < 0:
        return -1
    else:
        return 0

def point_path_side(point: tuple[int, int], path: list[tuple[int, int]]) -> int:
    """
    Determines which side of a path (series of connected line segments) a point is on.
    Uses the closest segment to determine the side.
    """
    if len(path) < 2:
        return 0
    
    min_distance = float('inf')
    closest_side = 0
    
    # Check each segment of the path
    for i in range(len(path) - 1):
        segment = (path[i], path[i + 1])
        
        # Calculate distance from point to this segment
        distance = point_to_line_distance(point, segment)
        
        # If this is the closest segment so far, use its side determination
        if distance < min_distance:
            min_distance = distance
            closest_side = point_line_side(point, segment)
    
    return closest_side

def point_to_line_distance(point: tuple[int, int], line_segment: tuple[tuple[int, int], tuple[int, int]]) -> float:
    """
    Calculates the distance from a point to a line segment.
    """
    px, py = point
    (x1, y1), (x2, y2) = line_segment
    
    # Calculate the distance from point to line segment
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    
    if len_sq == 0:  # Line segment is actually a point
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    
    param = dot / len_sq
    
    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D
    
    dx = px - xx
    dy = py - yy
    return (dx * dx + dy * dy) ** 0.5

def create_region_map(size: tuple[int, int], paths: list[list[tuple[int, int]]]) -> dict[tuple[int, ...], int]:
    """
    Creates a mapping from region signatures to region IDs.
    Each pixel's region is determined by which side of each path it's on.
    """
    width, height = size
    region_signatures = {}
    region_id = 0
    
    # Create a 2D array to store region IDs for each pixel
    region_map = {}
    
    for y in range(height):
        for x in range(width):
            # Create signature for this pixel based on which side of each path it's on
            signature = tuple(point_path_side((x, y), path) for path in paths)
            
            # If this signature hasn't been seen before, assign it a new region ID
            if signature not in region_signatures:
                region_signatures[signature] = region_id
                region_id += 1
            
            region_map[(x, y)] = region_signatures[signature]
    
    return region_map

def mesh_images_with_lines(images: list[Image.Image], size=(600, 400), num_cuts=3) -> Image.Image:
    """
    Divides canvas with randomized path cuts and fills each region with a different random image.
    Ensures no blank space by completely filling the canvas.
    """
    if len(images) < 2:
        raise ValueError("Need at least 2 images to mesh.")

    # Resize all to same size
    images = [img.resize(size).convert("RGB") for img in images]
    width, height = size

    # Generate random jagged/curved paths that cut across the canvas
    paths = generate_random_lines(size, num_cuts)
    
    # Create region map - each pixel gets assigned to a region
    region_map = create_region_map(size, paths)
    
    # Find all unique regions
    unique_regions = set(region_map.values())
    
    # Assign each region a random image
    region_to_image = {}
    for region_id in unique_regions:
        region_to_image[region_id] = random.choice(images)
    
    # Create output image
    output = Image.new("RGB", (width, height))
    
    # Fill each pixel based on its region
    for y in range(height):
        for x in range(width):
            region_id = region_map[(x, y)]
            source_img = region_to_image[region_id]
            
            # Get pixel from the assigned image for this region
            pixel = source_img.getpixel((x, y))
            output.putpixel((x, y), pixel)
    
    return output

# Keep backward compatibility
def mesh_images_chaotic(images: list[Image.Image], size=(600, 400)) -> Image.Image:
    """
    Backward compatibility wrapper - uses new randomized path-cutting algorithm.
    """
    return mesh_images_with_lines(images, size, num_cuts=random.randint(3, 6))

def get_random_images(count: int, size=(600, 400)) -> list[Image.Image]:
    """
    Fetches random images based on IMAGE_SOURCE_MODE configuration.
    Modes: 'cache_only', 'api_only', 'mixed'
    """
    # Clean up cache if needed
    cleanup_cache_if_needed()
    
    print(f"Image source mode: {IMAGE_SOURCE_MODE}")
    
    metadata = load_cache_metadata()
    cached_photo_ids = list(metadata.keys())
    
    images = []
    used_photo_ids = set()
    
    if IMAGE_SOURCE_MODE == 'cache_only':
        # Cache-only mode: Only use cached images
        if len(cached_photo_ids) < count:
            raise ValueError(f"Not enough cached images! Need {count}, have {len(cached_photo_ids)}. "
                           f"Run 'python cache_downloader.py {count}' to download more images.")
        
        selected_cached = random.sample(cached_photo_ids, count)
        print(f"Loading {count} images from cache (cache-only mode)...")
        
        for photo_id in selected_cached:
            try:
                img = load_image_from_cache(photo_id)
                img = img.resize(size).convert("RGB")
                images.append(img)
                used_photo_ids.add(photo_id)
            except Exception as e:
                print(f"Failed to load cached image {photo_id}: {e}")
                continue
                
    elif IMAGE_SOURCE_MODE == 'api_only':
        # API-only mode: Always fetch from API (but still save to cache)
        print(f"Fetching {count} images from Flickr API (api-only mode)...")
        
        # Use common parameters from config, with dynamic values
        params = FLICKR_SEARCH_PARAMS.copy()
        params['api_key'] = API_KEY
        params['page'] = random.randint(1, 10)  # Randomize page for variety
        
        # Request photos from Flickr
        response = requests.get(URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'photos' not in data or 'photo' not in data['photos']:
            raise ValueError("Failed to get photos from Flickr API")
        
        photos = data['photos']['photo']
        
        # Randomly select photos we need
        selected_photos = random.sample(photos, min(count, len(photos)))
        
        for photo in selected_photos:
            try:
                # Always download from API, even if cached
                photo_url = f"https://live.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}_c.jpg"
                
                # Download and convert to PIL Image
                img_response = requests.get(photo_url)
                img_response.raise_for_status()
                img = Image.open(BytesIO(img_response.content)).convert("RGB")
                
                # Cache the downloaded image
                save_image_to_cache(img, photo['id'], photo)
                print(f"Downloaded and cached image {photo['id']}")
                
                # Resize to requested size
                img = img.resize(size).convert("RGB")
                images.append(img)
                used_photo_ids.add(photo['id'])
                
            except Exception as e:
                print(f"Failed to download image {photo['id']}: {e}")
                continue
                
    else:  # 'mixed' mode (default)
        # Mixed mode: Use cache first, then API if needed
        cached_count = min(count, len(cached_photo_ids))
        if cached_count > 0:
            selected_cached = random.sample(cached_photo_ids, cached_count)
            print(f"Loading {len(selected_cached)} images from cache...")
            
            for photo_id in selected_cached:
                try:
                    img = load_image_from_cache(photo_id)
                    img = img.resize(size).convert("RGB")
                    images.append(img)
                    used_photo_ids.add(photo_id)
                except Exception as e:
                    print(f"Failed to load cached image {photo_id}: {e}")
                    continue
        
        # If we need more images, fetch from Flickr API
        remaining_count = count - len(images)
        if remaining_count > 0:
            print(f"Fetching {remaining_count} new images from Flickr API...")
            
            # Use common parameters from config, with dynamic values
            params = FLICKR_SEARCH_PARAMS.copy()
            params['api_key'] = API_KEY
            params['page'] = random.randint(1, 10)  # Randomize page for variety
            
            # Request photos from Flickr
            response = requests.get(URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'photos' not in data or 'photo' not in data['photos']:
                raise ValueError("Failed to get photos from Flickr API")
            
            photos = data['photos']['photo']
            
            # Filter out photos we've already used
            available_photos = [p for p in photos if p['id'] not in used_photo_ids]
            
            # Randomly select the remaining photos we need
            selected_photos = random.sample(available_photos, min(remaining_count, len(available_photos)))
            
            for photo in selected_photos:
                try:
                    # Check if this image is already cached (might have been added by another process)
                    if is_image_cached(photo['id']):
                        img = load_image_from_cache(photo['id'])
                        print(f"Found image {photo['id']} in cache")
                    else:
                        # Construct Flickr image URL
                        photo_url = f"https://live.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}_c.jpg"
                        
                        # Download and convert to PIL Image
                        img_response = requests.get(photo_url)
                        img_response.raise_for_status()
                        img = Image.open(BytesIO(img_response.content)).convert("RGB")
                        
                        # Cache the downloaded image
                        save_image_to_cache(img, photo['id'], photo)
                        print(f"Downloaded and cached image {photo['id']}")
                    
                    # Resize to requested size
                    img = img.resize(size).convert("RGB")
                    images.append(img)
                    used_photo_ids.add(photo['id'])
                    
                except Exception as e:
                    print(f"Failed to download image {photo['id']}: {e}")
                    continue
    
    if len(images) < count:
        print(f"Warning: Only got {len(images)} images instead of requested {count}")
    
    # Print cache stats
    cache_size = get_cache_size()
    cached_images_count = len(load_cache_metadata())
    print(f"Cache: {cached_images_count} images, {cache_size:.1f}MB")
    
    return images

if __name__ == "__main__":
    count = random.randint(2,5)
    count = 5
    imgs = get_random_images(count)
    result = mesh_images_chaotic(imgs)
    result.show()
