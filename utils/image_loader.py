from PIL import Image
import os
from typing import List, Tuple


def get_image_files(directory: str, supported_formats: List[str]) -> List[str]:
    """
    Get all image files from a directory.
    
    Args:
        directory: Path to directory containing images
        supported_formats: List of supported file extensions (e.g., ['.jpg', '.png'])
    
    Returns:
        List of full file paths to images
    """
    
    image_files = []
    
    with os.scandir(directory) as dir:
        
        for image_file_path in dir:
            extension = os.path.splitext(image_file_path)[1]
            if extension in supported_formats:
               image_files.append(str(image_file_path.path)) 
            else:
                print("Image format not supported!")
    
    return sorted(image_files)


def load_image(filepath: str) -> Image.Image:
    """
    Load an image from filepath and convert to RGB.
    
    Args:
        filepath: Path to image file
    
    Returns:
        PIL Image object in RGB mode
    """
    
    with Image.open(filepath) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.load()
        
    return img


def get_image_dimensions(image: Image.Image) -> Tuple[int, int]:
    """
    Get image dimensions.
    
    Args:
        image: PIL Image object
    
    Returns:
        Tuple of (width, height)
    """
    
    return image.size


def validate_image(filepath: str) -> bool:
    """
    Check if a file is a valid image.
    
    Args:
        filepath: Path to image file
    
    Returns:
        True if valid image, False otherwise
    """
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except:
        return False