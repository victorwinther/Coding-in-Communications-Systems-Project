import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import cv2
import scipy.fftpack

def rgb_to_ycbcr(image):
    """
    Convert an RGB image to YCbCr and return separate Y, Cb, and Cr images.
    Parameters: a numpy array representing the RGB image.
    Returns: Y, Cb, Cr: numpy arrays representing the Y, Cb, and Cr components.
    """
    img_array = np.array(image, dtype=float)
    
    # Separate the RGB channels
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    
    return Y, Cb, Cr

def save_component_image(component, filename):
    """
    Save a single Y, Cb, or Cr component as an image.
    
    Parameters:
    - component: a numpy array representing the Y, Cb, or Cr component.
    - filename: the name of the file to save the image as.
    """
    component_uint8 = np.uint8(component)
    iio.imwrite(filename, component_uint8)

def apply_420_subsampling(Cb, Cr):
    """
    Apply 4:2:0 subsampling to the Cb and Cr components.
    
    Parameters:
    - Cb: numpy array representing the Cb component.
    - Cr: numpy array representing the Cr component.
    
    Returns:
    - Cb_subsampled, Cr_subsampled: numpy arrays representing the subsampled Cb and Cr components.
    """
    # Perform 4:2:0 subsampling
    Cb_subsampled = Cb[::2, ::2]  # Take every second row and column
    Cr_subsampled = Cr[::2, ::2]  # Take every second row and column
    
    return Cb_subsampled, Cr_subsampled

def ycbcr_to_rgb(Y, Cb_subsampled, Cr_subsampled):
    """
    Convert YCbCr components back to RGB.
    
    Parameters:
    - Y: numpy array representing the Y component.
    - Cb_subsampled: numpy array representing the subsampled Cb component.
    - Cr_subsampled: numpy array representing the subsampled Cr component.
    
    Returns:
    - rgb_image: numpy array representing the reconstructed RGB image.
    """
    # Upsample Cb and Cr components to match Y size (repeat rows and columns)
    height, width = Y.shape
    Cb_upsampled = np.repeat(np.repeat(Cb_subsampled, 2, axis=0), 2, axis=1)
    Cr_upsampled = np.repeat(np.repeat(Cr_subsampled, 2, axis=0), 2, axis=1)
    
    # Perform inverse YCbCr to RGB conversion
    R = Y + 1.402 * (Cr_upsampled - 128)
    G = Y - 0.344136 * (Cb_upsampled - 128) - 0.714136 * (Cr_upsampled - 128)
    B = Y + 1.772 * (Cb_upsampled - 128)
    
    # Stack R, G, B channels and clip values to [0, 255]
    rgb_image = np.stack((R, G, B), axis=-1)
    rgb_image = np.clip(rgb_image, 0, 255)
    rgb_image = np.uint8(rgb_image)
    
    return rgb_image

# Example usage:
image_path = 'sample.bmp'
image = iio.imread(image_path)
Y, Cb, Cr = rgb_to_ycbcr(image)

# Save the Y, Cb, and Cr images
save_component_image(Y, 'Y_component.bmp')
save_component_image(Cb, 'Cb_component.bmp')
save_component_image(Cr, 'Cr_component.bmp')

# To visualize the components
plt.figure(figsize=(10, 3))

plt.subplot(1, 3, 1)
plt.imshow(Y, cmap='gray')
plt.title("Y Component")

plt.subplot(1, 3, 2)
plt.imshow(Cb, cmap='gray')
plt.title("Cb Component")

plt.subplot(1, 3, 3)
plt.imshow(Cr, cmap='gray')
plt.title("Cr Component")

plt.show()

Cb_subsampled, Cr_subsampled = apply_420_subsampling(Cb, Cr)

# Visualize the original and subsampled Cb and Cr components
plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.imshow(Cb, cmap='gray')
plt.title("Original Cb Component")

plt.subplot(1, 2, 2)
plt.imshow(Cb_subsampled, cmap='gray')
plt.title("Subsampled Cb Component")

plt.show()

plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.imshow(Cr, cmap='gray')
plt.title("Original Cr Component")

plt.subplot(1, 2, 2)
plt.imshow(Cr_subsampled, cmap='gray')
plt.title("Subsampled Cr Component")

plt.show()

# Convert YCbCr components back to RGB
reconstructed_rgb_image = ycbcr_to_rgb(Y, Cb_subsampled, Cr_subsampled)

# Display the reconstructed RGB image
plt.imshow(reconstructed_rgb_image)
plt.title("Reconstructed RGB Image")
plt.show()

# Convert using your function
Y, Cb, Cr = rgb_to_ycbcr(image)

# Convert using OpenCV
image_ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
Y_cv = image_ycbcr[:, :, 0]
Cb_cv = image_ycbcr[:, :, 1]
Cr_cv = image_ycbcr[:, :, 2]

# Plot the results for comparison
plt.figure(figsize=(15, 5))

plt.subplot(2, 3, 1)
plt.imshow(Y, cmap='gray')
plt.title("Y Component (Custom)")

plt.subplot(2, 3, 2)
plt.imshow(Cb, cmap='gray')
plt.title("Cb Component (Custom)")

plt.subplot(2, 3, 3)
plt.imshow(Cr, cmap='gray')
plt.title("Cr Component (Custom)")

plt.subplot(2, 3, 4)
plt.imshow(Y_cv, cmap='gray')
plt.title("Y Component (OpenCV)")

plt.subplot(2, 3, 5)
plt.imshow(Cb_cv, cmap='gray')
plt.title("Cb Component (OpenCV)")

plt.subplot(2, 3, 6)
plt.imshow(Cr_cv, cmap='gray')
plt.title("Cr Component (OpenCV)")

plt.show()

# Subsample using your function
Cb_subsampled, Cr_subsampled = apply_420_subsampling(Cb, Cr)

# Subsample using OpenCV (4:2:0 subsampling)
Cb_subsampled_cv = Cb_cv[::2, ::2]
Cr_subsampled_cv = Cr_cv[::2, ::2]

# Plot the results for comparison
plt.figure(figsize=(15, 5))

plt.subplot(2, 2, 1)
plt.imshow(Cb_subsampled, cmap='gray')
plt.title("Subsampled Cb (Custom)")

plt.subplot(2, 2, 2)
plt.imshow(Cr_subsampled, cmap='gray')
plt.title("Subsampled Cr (Custom)")

plt.subplot(2, 2, 3)
plt.imshow(Cb_subsampled_cv, cmap='gray')
plt.title("Subsampled Cb (OpenCV)")

plt.subplot(2, 2, 4)
plt.imshow(Cr_subsampled_cv, cmap='gray')
plt.title("Subsampled Cr (OpenCV)")

plt.show()

# Convert YCbCr components back to RGB
reconstructed_rgb_image = ycbcr_to_rgb(Y, Cb_subsampled, Cr_subsampled)

# Plot the reconstructed image and original image
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(reconstructed_rgb_image)
plt.title("Reconstructed RGB Image (Custom)")

plt.subplot(1, 2, 2)
plt.imshow(image)
plt.title("Original RGB Image")

plt.show()