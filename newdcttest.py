import numpy as np
import cv2
import scipy.fftpack
import matplotlib.pyplot as plt

def rgb_to_ycbcr(image):
    img_array = np.array(image, dtype=float)
    
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    
    ycbcr_image = np.stack((Y, Cb, Cr), axis=-1)
    
    return ycbcr_image

def split_into_blocks(image):
    h, w, _ = image.shape
    blocks = []
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = image[i:i+8, j:j+8, :]
            blocks.append(block)
    return blocks

def apply_dct(block):
    dct_block = np.zeros_like(block)
    for k in range(3):  # Anvend DCT på hver af de tre komponenter separat
        dct_block[:, :, k] = scipy.fftpack.dct(scipy.fftpack.dct(block[:, :, k].T, norm='ortho').T, norm='ortho')
    return dct_block

def quantize(block, q_matrix):
    quant_block = np.zeros_like(block)
    for k in range(3):  # Kvantiser hver af de tre komponenter separat
        quant_block[:, :, k] = np.round(block[:, :, k] / q_matrix).astype(int)
    return quant_block

quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def jpeg_compress(image_path):
    image = cv2.imread(image_path)
    image_ycbcr = rgb_to_ycbcr(image)
    blocks = split_into_blocks(image_ycbcr)
    
    dct_blocks = [apply_dct(block) for block in blocks]
    quantized_blocks = [quantize(block, quantization_matrix) for block in dct_blocks]
    
    return image, image_ycbcr, blocks, dct_blocks, quantized_blocks

def plot_image_steps(image_path):
    image, image_ycbcr, blocks, dct_blocks, quantized_blocks = jpeg_compress(image_path)
    
    # Plot original image
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    
    # Convert YCbCr back to RGB for display
    Y, Cb, Cr = cv2.split(image_ycbcr.astype(np.uint8))
    YCbCr_image = cv2.merge([Y, Cb, Cr])
    RGB_image = cv2.cvtColor(YCbCr_image, cv2.COLOR_YCrCb2RGB)
    
    # Plot YCbCr image
    plt.subplot(2, 3, 2)
    plt.imshow(RGB_image)
    plt.title("YCbCr Image")
    
    # Plot a sample block
    sample_block_index = 0  # Display the first block for simplicity
    plt.subplot(2, 3, 3)
    plt.imshow(blocks[sample_block_index].astype(np.uint8))
    plt.title("8x8 Block (Sample)")
    
    # Plot DCT of the sample block
    plt.subplot(2, 3, 4)
    plt.imshow(dct_blocks[sample_block_index][:, :, 0], cmap='gray')
    plt.title("DCT of Block (Sample, Y component)")
    
    # Plot Quantized DCT of the sample block
    plt.subplot(2, 3, 5)
    plt.imshow(quantized_blocks[sample_block_index][:, :, 0], cmap='gray')
    plt.title("Quantized DCT of Block (Sample, Y component)")
    
    plt.show()

# Call the function with your image path
plot_image_steps('pictures/mountain.png')
