import numpy as np
from PIL import Image, ImageTk
from scipy.fftpack import dct, idct
from heapq import heapify, heappop, heappush
from collections import Counter
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os
import tkinter as tk
from tkinter import filedialog, Scale, Label, Button, PhotoImage

# Helper functions for JPEG compression
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
    height, width = Y.shape
    Cb_upsampled = np.repeat(np.repeat(Cb_subsampled, 2, axis=0), 2, axis=1)
    Cr_upsampled = np.repeat(np.repeat(Cr_subsampled, 2, axis=0), 2, axis=1)
    R = Y + 1.402 * (Cr_upsampled - 128)
    G = Y - 0.344136 * (Cb_upsampled - 128) - 0.714136 * (Cr_upsampled - 128)
    B = Y + 1.772 * (Cb_upsampled - 128)
    rgb_image = np.stack((R, G, B), axis=-1)
    rgb_image = np.clip(rgb_image, 0, 255)
    rgb_image = np.uint8(rgb_image)
    rgb_image_pil = Image.fromarray(rgb_image)
    return rgb_image_pil

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
    reconstructed_image.save("reconstructed_image.jpg", "JPEG", quality=95)
    print("Det rekonstruerede billede er gemt som 'reconstructed_image.jpg'")
    reconstructed_image.save("reconstructed_image.png", "PNG")
    print("Det rekonstruerede billede er gemt som 'reconstructed_image.png'")
    reconstructed_jpeg_size = os.path.getsize('reconstructed_image.jpg')
    print(f"Rekonstrueret JPEG-fil størrelse: {reconstructed_jpeg_size / 1024:.2f} KB")
    reconstructed_png_size = os.path.getsize('reconstructed_image.png')
    print(f"Rekonstrueret PNG-fil størrelse: {reconstructed_png_size / 1024:.2f} KB")

# Default quantization tables
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

# GUI code
class JPEGCompressionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JPEG Compression Tool")
        
        self.label = Label(root, text="Choose an image and adjust the quality factor:")
        self.label.pack()
        
        self.upload_button = Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()
        
        self.scale = Scale(root, from_=1, to=100, orient='horizontal', label='Quality Factor', command=self.update_image)
        self.scale.pack()
        
        self.image_label = Label(root)
        self.image_label.pack()
        
        self.stats_label = Label(root, text="")
        self.stats_label.pack()
        
        self.image_path = None
        
    def upload_image(self):
        self.image_path = filedialog.askopenfilename()
        self.update_image()

    def update_image(self, event=None):
        if self.image_path:
            qf = self.scale.get()
            reconstructed_image, encoded_data = jpeg_compression(self.image_path, qf)
            self.display_image(reconstructed_image)
            self.display_statistics(reconstructed_image, encoded_data)
    
    def display_image(self, image):
        img_resized = image.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img_resized)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk
    
    def display_statistics(self, reconstructed_image, encoded_data):
        stats = f"Quality Factor: {self.scale.get()}\n"
        image = iio.imread(self.image_path)
        entropy_mountain = Entropy(image)
        decoded_image = np.array(reconstructed_image)
        stats += f'Lower limit for average code length per pixel: {entropy_mountain:.2f} bits\n'
        stats += f'Average code length per pixel: {len(encoded_data) / image.size:.2f} bits\n'
        stats += f'Entropy: {Entropy(decoded_image):.2f} bits\n'
        stats += f'MSE: {MSE(image, decoded_image):.2f}\n'
        stats += f'PSNR: {PSNR(image, decoded_image):.2f} dB\n'
        self.stats_label.config(text=stats)

# Run the GUI
root = tk.Tk()
gui = JPEGCompressionGUI(root)
root.mainloop()
