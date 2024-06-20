import numpy as np
from PIL import Image

def rgb_to_ycbcr(image):
    """
    Convert an RGB image to YCbCr.
    
    Parameters:
    - image: a PIL Image object in RGB mode.
    
    Returns:
    - ycbcr_image: a numpy array representing the image in YCbCr color space.
    """
    # Convert the image to numpy array
    img_array = np.array(image, dtype=float)
    
    # Separate the RGB channels
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]
    
    # Apply the conversion formulas
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    
    # Stack the channels back together
    ycbcr_image = np.stack((Y, Cb, Cr), axis=-1)
    
    return ycbcr_image

# Example usage:
image_path = 'sample.bmp'
image = Image.open(image_path).convert('RGB')
ycbcr_image = rgb_to_ycbcr(image)

# To visualize the result
ycbcr_image_pil = Image.fromarray(np.uint8(ycbcr_image))
ycbcr_image_pil.show()
