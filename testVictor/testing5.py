import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
from heapq import heapify, heappop, heappush
from collections import Counter
import matplotlib.pyplot as plt
import imageio.v3 as iio
import cv2
import os

def rgb_to_ycbcr(image):
    img_array = np.array(image, dtype=float)
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    return Y, Cb, Cr

def apply_420_subsampling(Cb, Cr):
    Cb_subsampled = Cb[::2, ::2]
    Cr_subsampled = Cr[::2, ::2]
    return Cb_subsampled, Cr_subsampled


def ycbcr_to_rgb(Y, Cb_subsampled, Cr_subsampled):
    # Ensure inputs are in floating point
    Y = Y.astype(np.float32)
    Cb_subsampled = Cb_subsampled.astype(np.float32)
    Cr_subsampled = Cr_subsampled.astype(np.float32)

    height, width = Y.shape
    Cb_upsampled = np.repeat(np.repeat(Cb_subsampled, 2, axis=0), 2, axis=1)
    Cr_upsampled = np.repeat(np.repeat(Cr_subsampled, 2, axis=0), 2, axis=1)
    Cb_upsampled = Cb_upsampled[:height, :width]
    Cr_upsampled = Cr_upsampled[:height, :width]

    R = Y + 1.402 * (Cr_upsampled - 128)
    G = Y - 0.344136 * (Cb_upsampled - 128) - 0.714136 * (Cr_upsampled - 128)
    B = Y + 1.772 * (Cb_upsampled - 128)

    rgb_image = np.stack((R, G, B), axis=-1)
    rgb_image = np.clip(rgb_image, 0, 255)
    rgb_image = np.uint8(rgb_image)
    rgb_image_pil = Image.fromarray(rgb_image)

    return rgb_image_pil

def ycbcr_to_rgb_test(Y, Cb, Cr):
    Y = Y.astype(np.float32)
    Cb = Cb.astype(np.float32)
    Cr = Cr.astype(np.float32)
    
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)

    rgb_image = np.stack((R, G, B), axis=-1)
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
    return rgb_image

def scale_quantization_matrices(qf, quant_table_luminance, quant_table_chrominance):
    if qf < 50:
        q_scale = max(1, 5000 / qf)
    else:
        q_scale = 200 - 2 * qf
    scaled_luminance = np.floor((quant_table_luminance * q_scale + 50) / 100)
    scaled_chrominance = np.floor((quant_table_chrominance * q_scale + 50) / 100)
    scaled_luminance[scaled_luminance == 0] = 1
    scaled_chrominance[scaled_chrominance == 0] = 1
    return scaled_luminance, scaled_chrominance

def quantize_dct(dct_block, quant_matrix):
    return np.round(dct_block / quant_matrix)

def dequantize_dct(dct_block, quant_matrix):
    return dct_block * quant_matrix

def dct2d_library(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2d_library(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def block_process(channel, block_size, process_block, quant_matrix):
    h, w = channel.shape[:2]
    blocks = (channel.reshape(h // block_size, block_size, -1, block_size)
                      .swapaxes(1, 2)
                      .reshape(-1, block_size, block_size))
    processed_blocks = np.array([process_block(block) for block in blocks])
    quantized_blocks = np.array([quantize_dct(block, quant_matrix) for block in processed_blocks])
    return (quantized_blocks.reshape(h // block_size, w // block_size, block_size, block_size)
                            .swapaxes(1, 2)
                            .reshape(h, w))

def block_process_inverse(channel, block_size, process_block, quant_matrix):
    h, w = channel.shape[:2]
    blocks = (channel.reshape(h // block_size, block_size, -1, block_size)
                      .swapaxes(1, 2)
                      .reshape(-1, block_size, block_size))
    dequantized_blocks = np.array([dequantize_dct(block, quant_matrix) for block in blocks])
    processed_blocks = np.array([process_block(block) for block in dequantized_blocks])
    return (processed_blocks.reshape(h // block_size, w // block_size, block_size, block_size)
                            .swapaxes(1, 2)
                            .reshape(h, w))

def dct2d_manual(block):
    N = block.shape[0]
    dct_matrix = np.zeros((N, N))

    def alpha(u):
        return np.sqrt(1/2) if u == 0 else 1

    for u in range(N):
        for v in range(N):
            sum_value = 0.0
            for x in range(N):
                for y in range(N):
                    sum_value += block[x, y] * np.cos((2 * x + 1) * u * np.pi / (2 * N)) * np.cos((2 * y + 1) * v * np.pi / (2 * N))
            dct_matrix[u, v] = 0.25 * alpha(u) * alpha(v) * sum_value

    return dct_matrix

def idct2d_manual(block):
    N = block.shape[0]
    idct_matrix = np.zeros((N, N))

    def alpha(u):
        return np.sqrt(1/2) if u == 0 else 1

    for x in range(N):
        for y in range(N):
            sum_value = 0.0
            for u in range(N):
                for v in range(N):
                    sum_value += alpha(u) * alpha(v) * block[u, v] * np.cos((2 * x + 1) * u * np.pi / (2 * N)) * np.cos((2 * y + 1) * v * np.pi / (2 * N))
            idct_matrix[x, y] = 0.25 * sum_value

    return idct_matrix


def zigzag_scan(block):
    zigzag_index = np.array([
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63]
    ]).flatten()
    return block.flatten()[zigzag_index]

def run_length_encode(arr):
    result = []
    count = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1]:
            count += 1
        else:
            result.append((arr[i - 1], count))
            count = 1
    result.append((arr[-1], count))
    return result

def run_length_decode(rle):
    result = []
    for value, count in rle:
        result.extend([value] * count)
    return np.array(result)

def zigzag_decode(arr):
    zigzag_index = np.array([
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63]
    ]).flatten()
    block = np.zeros((8, 8))
    block.flatten()[zigzag_index] = arr
    return block


class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in freq.items()]
    heapify(heap)
    while len(heap) > 1:
        node1 = heappop(heap)
        node2 = heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heappush(heap, merged)
    return heap[0]

def generate_huffman_codes(node, prefix='', codebook=None):
    if codebook is None:
        codebook = {}
    if node is not None:
        if node.symbol is not None:
            codebook[node.symbol] = prefix
        generate_huffman_codes(node.left, prefix + '0', codebook)
        generate_huffman_codes(node.right, prefix + '1', codebook)
    return codebook

def huffman_encode(image):
    flattened_image = image.flatten()
    freq = Counter(flattened_image)
    huffman_tree = build_huffman_tree(freq)
    huffman_codes = generate_huffman_codes(huffman_tree)
    encoded_image = ''.join([huffman_codes[int(pixel)] for pixel in flattened_image])
    return encoded_image, huffman_codes

def huffman_decode(encoded_image, huffman_codes):
    reverse_huff_dict = {v: k for k, v in huffman_codes.items()}
    current_code = ""
    decoded_image = []
    for bit in encoded_image:
        current_code += bit
        if current_code in reverse_huff_dict:
            decoded_image.append(reverse_huff_dict[current_code])
            current_code = ""
    return np.array(decoded_image)

def reshape_decoded_data(decoded_data, Y_shape, Cb_shape, Cr_shape):
    Y_size = np.prod(Y_shape)
    Cb_size = np.prod(Cb_shape)
    Cr_size = np.prod(Cr_shape)
    Y_decoded = decoded_data[:Y_size].reshape(Y_shape)
    Cb_decoded = decoded_data[Y_size:Y_size + Cb_size].reshape(Cb_shape)
    Cr_decoded = decoded_data[Y_size + Cb_size:].reshape(Cr_shape)

    
    return Y_decoded, Cb_decoded, Cr_decoded

def Entropy(im):
    histogram, bin_edges = np.histogram(im, bins=range(256))
    p = histogram / np.sum(histogram)
    p1 = p[p != 0]
    entropy = -np.dot(p1.T, np.log2(p1))
    return entropy

def MSE(im1, im2):
    return np.mean((im1 - im2) ** 2)

def PSNR(im1, im2):
    mse = MSE(im1, im2)
    if mse == 0:
        return float('inf')
    max_pixel = 2 ** 8 - 1
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def jpeg_compression(image_path, qf):
    image = Image.open(image_path).convert('RGB')
    Y, Cb, Cr = rgb_to_ycbcr(image)
    Cb_subsampled, Cr_subsampled = apply_420_subsampling(Cb, Cr)
    scaled_luminance, scaled_chrominance = scale_quantization_matrices(qf, quant_table_luminance, quant_table_chrominance)
    Y_dct = block_process(Y, 8, dct2d_library, scaled_luminance)
    Cb_dct = block_process(Cb_subsampled, 8, dct2d_library, scaled_chrominance)
    Cr_dct = block_process(Cr_subsampled, 8, dct2d_library, scaled_chrominance)
    all_dct_coeffs = np.concatenate((Y_dct.flatten(), Cb_dct.flatten(), Cr_dct.flatten())).astype(int)
    encoded_data, huff_dict = huffman_encode(all_dct_coeffs)
    decoded_data = huffman_decode(encoded_data, huff_dict)
    Y_decoded, Cb_decoded, Cr_decoded = reshape_decoded_data(decoded_data, Y_dct.shape, Cb_dct.shape, Cr_dct.shape)
    Y_reconstructed = block_process_inverse(Y_decoded, 8, idct2d_library, scaled_luminance)
    Cb_reconstructed = block_process_inverse(Cb_decoded, 8, idct2d_library, scaled_chrominance)
    Cr_reconstructed = block_process_inverse(Cr_decoded, 8, idct2d_library, scaled_chrominance)
    reconstructed_rgb_image = ycbcr_to_rgb(Y_reconstructed, Cb_reconstructed, Cr_reconstructed)
    return reconstructed_rgb_image, encoded_data

def print_statistics(image_path, reconstructed_image, encoded_data):
    image = iio.imread(image_path)
    entropy_mountain = Entropy(image)
    print(f'Nedre grænse for gennemsnitlig kodelængde per pixel: {entropy_mountain:.2f} bits')
    decoded_image = np.array(reconstructed_image)
    print(f'Gennemsnitlig kodelængde per pixel: {len(encoded_data) / image.size:.2f} bits')
    print(f'Entropi: {Entropy(decoded_image):.2f} bits')
    print(f'MSE: {MSE(image, decoded_image):.2f}')
    print(f'PSNR: {PSNR(image, decoded_image):.2f} dB')
    original_pixel_count = image.size
    print(f"Antal pixels før kodning: {original_pixel_count}")
    new_pixel_count = decoded_image.size
    print(f"Antal pixels efter kodning: {new_pixel_count}")
    encoded_bit_count = len(encoded_data)
    print(f"Antal bits efter kodning: {encoded_bit_count}")
    compression_ratio = original_pixel_count * 8 / encoded_bit_count
    print(f"Kompressionsforhold: {compression_ratio:.2f}")
    original_file_size = os.path.getsize(image_path)
    print(f"Original BMP-fil størrelse: {original_file_size / 1024:.2f} KB")
    encoded_data_size = len(encoded_data) / 8
    print(f"Huffman-kodet data størrelse: {encoded_data_size / 1024:.2f} KB")
    print("Det rekonstruerede billede er gemt som 'reconstructed_image.jpg'")
    reconstructed_image.save("reconstructed_image.png", "PNG")
    print("Det rekonstruerede billede er gemt som 'reconstructed_image.png'")
    reconstructed_jpeg_size = os.path.getsize('reconstructed_image.jpg')
    print(f"Rekonstrueret JPEG-fil størrelse: {reconstructed_jpeg_size / 1024:.2f} KB")
    reconstructed_png_size = os.path.getsize('reconstructed_image.png')
    print(f"Rekonstrueret PNG-fil størrelse: {reconstructed_png_size / 1024:.2f} KB")

quant_table_luminance = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

quant_table_chrominance = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

def calculate_entropy(bitstream):
    # Convert the bitstream (string) into a list of bits
    bit_array = list(bitstream)
    
    # Count the frequency of each bit (0 and 1)
    freq = Counter(bit_array)
    
    # Calculate probabilities
    total_bits = len(bit_array)
    probabilities = [freq[bit] / total_bits for bit in freq]
    
    # Calculate entropy
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy

# Definer kvantiseringsfaktorer
quant_factors = [10,30, 50, 75,100]

# Gem resultater for PSNR og entropi
psnrs = []
entropies = []

# Kør JPEG kompression for forskellige kvantiseringsfaktorer

for qf in quant_factors:
    reconstructed_image, encoded_data = jpeg_compression('pictures/parrots.bmp', qf)
    #plt.imshow(reconstructed_image)
    #plt.title(f'JPEG Compression with QF={qf}')
    #plt.show()
    image = iio.imread('pictures/parrots.bmp')
    psnrs.append(PSNR(image, np.array(reconstructed_image)))
    entropies.append(calculate_entropy(encoded_data))

# Lav rate-distortion plot
plt.figure(figsize=(10, 6))
plt.plot(entropies, psnrs, marker='o')
plt.title('Rate-Distortion Plot')
plt.xlabel('Entropy (bits/pixel)')
plt.ylabel('PSNR (dB)')
plt.grid(True)
plt.show()

# Definer kvantiseringsfaktoren
qf = 100 # Du kan justere denne værdi mellem 0 og 100

# Komprimer billedet og få rekonstrueret billede og kodet data
reconstructed_image, encoded_data = jpeg_compression('pictures/parrots.bmp', qf)
original_image = iio.imread('pictures/parrots.bmp')
 
# Display the original and reconstructed images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image)
plt.title('Reconstructed Image')
plt.axis('off')

plt.show()

# Udskriv statistik
print_statistics('pictures/parrots.bmp', reconstructed_image, encoded_data)
def test_color_transformation(image_path):
    # Load the image
    #original_image = Image.open(image_path).convert('RGB')
    original_image = iio.imread(image_path)
    # Convert RGB to YCbCr
    Y, Cb, Cr = rgb_to_ycbcr(original_image)
    
    # Convert YCbCr back to RGB
    reconstructed_image = ycbcr_to_rgb_test(Y, Cb, Cr)
    
    # Convert images to numpy arrays for MSE and PSNR calculation
    original_image_array = np.array(original_image)
    reconstructed_image_array = np.array(reconstructed_image)
    
    # Calculate MSE and PSNR
    mse_value = MSE(original_image_array, reconstructed_image_array)
    psnr_value = PSNR(original_image_array, reconstructed_image_array)
    
    # Display the original and reconstructed images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image)
    plt.title('Reconstructed Image')
    plt.axis('off')
    
    plt.show()
    
    # Print MSE and PSNR values
    print(f'MSE: {mse_value:.4f}')
    print(f'PSNR: {psnr_value:.2f} dB')

# Example usage
#test_color_transformation('pictures/I25.png')

def test_subsampling(image_path):
    # Load the image
    #original_image = Image.open(image_path).convert('RGB')
    original_image = iio.imread(image_path)
    # Convert RGB to YCbCr
    Y, Cb, Cr = rgb_to_ycbcr(original_image)

    Cb_subsampled, Cr_subsampled = apply_420_subsampling(Cb, Cr)
    reconstructed_image = ycbcr_to_rgb(Y, Cb_subsampled, Cr_subsampled)
    
    # Convert images to numpy arrays for MSE and PSNR calculation
    original_image_array = np.array(original_image)
    reconstructed_image_array = np.array(reconstructed_image)
    
    # Calculate MSE and PSNR
    mse_value = MSE(original_image_array, reconstructed_image_array)
    psnr_value = PSNR(original_image_array, reconstructed_image_array)
    
    # Display the original and reconstructed images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image)
    plt.title('Reconstructed Image')
    plt.axis('off')
    
    plt.show()
    
    # Print MSE and PSNR values
    print(f'MSE: {mse_value:.4f}')
    print(f'PSNR: {psnr_value:.2f} dB')
#test_subsampling('pictures/I25.png')    

def test_DCT(image_path):
    # Load the image
    #original_image = Image.open(image_path).convert('RGB')
    original_image = iio.imread(image_path)
    # Convert RGB to YCbCr
    Y, Cb, Cr = rgb_to_ycbcr(original_image)

    Cb_subsampled, Cr_subsampled = apply_420_subsampling(Cb, Cr)
    scaled_luminance, scaled_chrominance = scale_quantization_matrices(100, quant_table_luminance, quant_table_chrominance)
    Y_dct = block_process(Y, 8, dct2d_library, scaled_luminance)
    Cb_dct = block_process(Cb_subsampled, 8, dct2d_library, scaled_chrominance)
    Cr_dct = block_process(Cr_subsampled, 8, dct2d_library, scaled_chrominance)
    all_dct_coeffs = np.concatenate((Y_dct.flatten(), Cb_dct.flatten(), Cr_dct.flatten())).astype(int)

    Y_decoded, Cb_decoded, Cr_decoded = reshape_decoded_data(all_dct_coeffs, Y_dct.shape, Cb_dct.shape, Cr_dct.shape)
    Y_reconstructed = block_process_inverse(Y_decoded, 8, idct2d_library, scaled_luminance)
    Cb_reconstructed = block_process_inverse(Cb_decoded, 8, idct2d_library, scaled_chrominance)
    Cr_reconstructed = block_process_inverse(Cr_decoded, 8, idct2d_library, scaled_chrominance)
    reconstructed_image = ycbcr_to_rgb(Y_reconstructed, Cb_reconstructed, Cr_reconstructed)
    
    
    # Convert images to numpy arrays for MSE and PSNR calculation
    original_image_array = np.array(original_image)
    reconstructed_image_array = np.array(reconstructed_image)
    
    # Calculate MSE and PSNR
    mse_value = MSE(original_image_array, reconstructed_image_array)
    psnr_value = PSNR(original_image_array, reconstructed_image_array)
    
    # Display the original and reconstructed images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image)
    plt.title('Reconstructed Image')
    plt.axis('off')
    
    plt.show()
    
    # Print MSE and PSNR values
    print(f'MSE: {mse_value:.4f}')
    print(f'PSNR: {psnr_value:.2f} dB')
#test_DCT('pictures/I25.png')    


def test_huffman_coding(image_path):
    # Load the image and convert it to grayscale for simplicity
    image = iio.imread(image_path)
    image_array = np.array(image)

    # Flatten the image array for Huffman encoding
    flattened_image = image_array.flatten()

    # Original data size
    original_size = flattened_image.size * flattened_image.itemsize * 8  # in bits


    Y, Cb, Cr = rgb_to_ycbcr(image)
    Cb_subsampled, Cr_subsampled = apply_420_subsampling(Cb, Cr)
    scaled_luminance, scaled_chrominance = scale_quantization_matrices(qf, quant_table_luminance, quant_table_chrominance)
    Y_dct = block_process(Y, 8, dct2d_manual, scaled_luminance)
    Cb_dct = block_process(Cb_subsampled, 8, dct2d_manual, scaled_chrominance)
    Cr_dct = block_process(Cr_subsampled, 8, dct2d_manual, scaled_chrominance)

    all_dct_coeffs = np.concatenate((Y_dct.flatten(), Cb_dct.flatten(), Cr_dct.flatten())).astype(int)
    encoded_data, huff_dict = huffman_encode(all_dct_coeffs)
    decoded_data = huffman_decode(encoded_data, huff_dict)


    Y_decoded, Cb_decoded, Cr_decoded = reshape_decoded_data(decoded_data, Y_dct.shape, Cb_dct.shape, Cr_dct.shape)
    Y_reconstructed = block_process_inverse(Y_decoded, 8, idct2d_manual, scaled_luminance)
    Cb_reconstructed = block_process_inverse(Cb_decoded, 8, idct2d_manual, scaled_chrominance)
    Cr_reconstructed = block_process_inverse(Cr_decoded, 8, idct2d_manual, scaled_chrominance)
    reconstructed_rgb_image = ycbcr_to_rgb(Y_reconstructed, Cb_reconstructed, Cr_reconstructed)   
    # Encoded data size
    encoded_size = len(encoded_data)  # in bits
    
    # Reshape decoded data to original shape
    decoded_image_array = np.array(reconstructed_rgb_image)
    # Check bitwise accuracy

    Y1_decoded, Cb1_decoded, Cr1_decoded = reshape_decoded_data(all_dct_coeffs, Y_dct.shape, Cb_dct.shape, Cr_dct.shape)
    Y1_reconstructed = block_process_inverse(Y1_decoded, 8, idct2d_library, scaled_luminance)
    Cb1_reconstructed = block_process_inverse(Cb1_decoded, 8, idct2d_library, scaled_chrominance)
    Cr1_reconstructed = block_process_inverse(Cr1_decoded, 8, idct2d_library, scaled_chrominance)
    reconstructed_rgb_image1 = ycbcr_to_rgb(Y1_reconstructed, Cb1_reconstructed, Cr1_reconstructed)

    bitwise_accuracy = np.array_equal(np.array(reconstructed_rgb_image1), decoded_image_array)
    # Calculate size after decoding
    decoded_size = decoded_image_array.size * decoded_image_array.itemsize * 8  # in bits

    # Display sizes and bitwise accuracy
    print(f'Original size: {original_size} bits')
    print(f'Encoded size: {encoded_size} bits')
    print(f'Decoded size: {decoded_size} bits')
    print(f'Bitwise accuracy: {"Pass" if bitwise_accuracy else "Fail"}')
    #print compression
    print(f'Compression ratio: {original_size / encoded_size:.2f}')

    # Display original and decoded images for visual inspection
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_array, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(decoded_image_array, cmap='gray')
    plt.title('Decoded Image')
    plt.axis('off')

    plt.show()

# Example usage
#test_huffman_coding('pictures/parrots.bmp')



import numpy as np
import cv2
import os
from PIL import Image

def convert_to_jpg(image_path, output_path, quality=95):
    # Load image using PIL
    image = Image.open(image_path)
    # Save image with specified JPEG quality
    image.save(output_path, 'JPEG', quality=quality)

def calculate_psnr(original_image, compressed_image):
    # Calculate MSE first
    mse = np.mean((original_image - compressed_image) ** 2)
    if mse == 0:
        return float('inf')
    # Calculate PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def test_jpg_conversion(image_path, quality_levels):
    original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    psnrs = []
    sizes = []
    for quality in quality_levels:
        # Define output path for the compressed image
        output_path = f'compressed_quality_{quality}.jpg'
        # Convert to JPG
        convert_to_jpg(image_path, output_path, quality=quality)
        # Read the compressed image
        compressed_image = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
        # Calculate PSNR
        psnr = calculate_psnr(original_image, compressed_image)
        psnrs.append(psnr)
        # Get file size
        file_size = os.path.getsize(output_path)
        sizes.append(file_size)
        # Optionally delete the file to save space
        os.remove(output_path)
        print(f'PSNR for quality {quality}: {psnr:.2f} dB, Size: {file_size} bytes')
    return psnrs, sizes

# Example usage
quality_levels = [10, 30, 50, 70, 90, 100]
psnrs, sizes = test_jpg_conversion('pictures/parrots.bmp', quality_levels)

import numpy as np
import os
import cv2
from PIL import Image

# Assuming jpeg_compression and auxiliary functions are already defined elsewhere in your code.

def test_custom_jpeg_compression(image_path, quality_levels):
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency with PIL

    psnrs = []
    sizes = []

    for quality in quality_levels:
        # Compress the image using your custom method
        reconstructed_image, encoded_data = jpeg_compression(image_path, quality)
        
        # Save the reconstructed image to measure file size
        output_path = f'custom_compressed_quality_{quality}.png'
        reconstructed_image.save(output_path, 'PNG')  # Save as PNG to avoid additional JPEG compression
        encoded_data_size = len(encoded_data) / 8
        print(f"Huffman-kodet data størrelse: {encoded_data_size / 1024:.2f} KB")
        
        # Calculate PSNR
        reconstructed_image_array = np.array(reconstructed_image)
        psnr = calculate_psnr(original_image, reconstructed_image_array)
        psnrs.append(psnr)
        
        # Get file size
        file_size = os.path.getsize(output_path)
        sizes.append(file_size)
        
        # Optionally delete the file to save space
        os.remove(output_path)
        
        print(f'PSNR for quality {quality}: {psnr:.2f} dB, Size: {encoded_data_size} bytes')
    
    return psnrs, sizes

def calculate_psnr(original_image, compressed_image):
    mse = np.mean((original_image - compressed_image) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Example usage
quality_levels = [10, 30, 50, 70, 90, 100]
psnrs, sizes = test_custom_jpeg_compression('pictures/parrots.bmp', quality_levels)

