import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
from heapq import heapify, heappop, heappush
from collections import Counter
import matplotlib.pyplot as plt
import imageio.v3 as iio

def calculate_entropy(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0,256])
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    return entropy


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

def quantize_dct(dct_block, quant_matrix):
    return np.round(dct_block / quant_matrix)

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

def dct2d_library(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

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

def generate_huffman_codes(node, prefix='', codebook={}):
    if node is not None:
        if node.symbol is not None:
            codebook[node.symbol] = prefix
        generate_huffman_codes(node.left, prefix + '0', codebook)
        generate_huffman_codes(node.right, prefix + '1', codebook)
    return codebook

def huffman_encoding(data):
    freq = Counter(data)
    huffman_tree = build_huffman_tree(freq)
    huffman_codes = generate_huffman_codes(huffman_tree)
    encoded_data = ''.join(huffman_codes[symbol] for symbol in data)
    return encoded_data, huffman_codes

def huffman_decoding(encoded_data, huffman_codes):
    reverse_huffman_codes = {v: k for k, v in huffman_codes.items()}
    decoded_data = []
    current_code = ''
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_huffman_codes:
            decoded_data.append(reverse_huffman_codes[current_code])
            current_code = ''
    return decoded_data

def run_length_encode(data):
    encoded = []
    prev_symbol = data[0]
    count = 1
    for symbol in data[1:]:
        if symbol == prev_symbol:
            count += 1
        else:
            encoded.append((prev_symbol, count))
            prev_symbol = symbol
            count = 1
    encoded.append((prev_symbol, count))
    return encoded

def run_length_decode(encoded):
    decoded = []
    for symbol, count in encoded:
        decoded.extend([symbol] * count)
    return decoded

def calculate_image_size(image_path):
    image = Image.open(image_path).convert('RGB')
    W, H = image.size
    size_start = W * H * 3  # RGB format
    return size_start

def count_nonzero_elements(array):
    return np.count_nonzero(array)

def calculate_quantized_size(Y_dct, Cb_dct, Cr_dct):
    Y_nonzero = count_nonzero_elements(Y_dct)
    Cb_nonzero = count_nonzero_elements(Cb_dct)
    Cr_nonzero = count_nonzero_elements(Cr_dct)
    total_nonzero = Y_nonzero + Cb_nonzero + Cr_nonzero
    return total_nonzero

def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    Y, Cb, Cr = rgb_to_ycbcr(image)
    Cb_subsampled, Cr_subsampled = apply_420_subsampling(Cb, Cr)

    Y_dct = block_process(Y, 8, dct2d_library, quant_table_luminance)
    Cb_dct = block_process(Cb_subsampled, 8, dct2d_library, quant_table_chrominance)
    Cr_dct = block_process(Cr_subsampled, 8, dct2d_library, quant_table_chrominance)

    return Y_dct, Cb_dct, Cr_dct

def compare_sizes(image_path):
    size_start = calculate_image_size(image_path)
    Y_dct, Cb_dct, Cr_dct = process_image(image_path)
    size_quantized = calculate_quantized_size(Y_dct, Cb_dct, Cr_dct)

    # Flatten DCT coefficients to a list
    all_dct_coeffs = np.concatenate((Y_dct.flatten(), Cb_dct.flatten(), Cr_dct.flatten())).astype(int)
    # Encode with run-length encoding
    run_length_encoded = run_length_encode(all_dct_coeffs)

    # Flatten run-length encoded data for Huffman encoding
    run_length_symbols = [symbol for symbol, count in run_length_encoded for _ in range(count)]

    # Perform Huffman encoding
    encoded_data, huff_dict = huffman_encoding(run_length_symbols)
    size_huffman = len(encoded_data)  # Size in bits

    # Perform Huffman decoding
    decoded_data = huffman_decoding(encoded_data, huff_dict)

    # Convert the decoded symbols back to run-length encoded format
    run_length_decoded_symbols = []
    symbol_index = 0
    for symbol, count in run_length_encoded:
        decoded_symbols = decoded_data[symbol_index:symbol_index + count]
        run_length_decoded_symbols.extend([(symbol, count)] * len(decoded_symbols))
        symbol_index += count

    # Decode run-length encoding
    run_length_decoded = run_length_decode(run_length_decoded_symbols)
    decoded_array = np.array(run_length_decoded[:len(all_dct_coeffs)]).reshape(Y_dct.shape)

    print(f"Original image size: {size_start} bytes")
    print(f"Size after quantization: {size_quantized} non-zero coefficients")
    print(f"Size after Huffman coding: {size_huffman // 8} bytes")

    entropy_original = Entropy(np.array(Image.open(image_path).convert('L')))
    entropy_compressed = Entropy(decoded_array)

    mse = MSE(np.array(Image.open(image_path).convert('L')), decoded_array)
    psnr = PSNR(np.array(Image.open(image_path).convert('L')), decoded_array)

    print(f'Entropy of original image: {entropy_original}')
    print(f'Entropy of compressed image: {entropy_compressed}')
    print(f'MSE: {mse}')
    print(f'PSNR: {psnr}')

def Entropy(im):
    histogram, bin_edges = np.histogram(im, bins=range(256))
    p = histogram / np.sum(histogram)
    p1 = p[p!=0]
    entropy = -np.dot(p1.T,np.log2(p1))
    return entropy

def MSE(im1,im2):
    return np.mean((im1-im2)**2)

def PSNR(im1,im2):
    mse = MSE(im1,im2)
    if mse == 0:
        return float('inf')
    max_pixel = 2**8-1
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Quantization tables
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

# Eksempel brug:
image_path = 'sample.bmp'
image1 = iio.imread(image_path);
#compare_sizes(image_path)
entropy_mountain = calculate_entropy(image1)
print(f'Nedre grænse for gennemsnitlig kodelængde per pixel: {entropy_mountain:.2f} bits')
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

def generate_huffman_codes(node, prefix='', codebook={}):
    if node is not None:
        if node.symbol is not None:
            codebook[node.symbol] = prefix
        generate_huffman_codes(node.left, prefix + '0', codebook)
        generate_huffman_codes(node.right, prefix + '1', codebook)
    return codebook

def huffman_encode(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0,256])
    freq = {i: hist[i] for i in range(256) if hist[i] > 0}
    huffman_tree = build_huffman_tree(freq)
    huffman_codes = generate_huffman_codes(huffman_tree)
    encoded_image = ''.join([huffman_codes[pixel] for pixel in image.flatten()])
    return encoded_image, huffman_codes

encoded_mountain, huffman_codes_mountain = huffman_encode(image1)

hist, _ = np.histogram(image1.flatten(), bins=256, range=[0,256])
freq = {i: hist[i] for i in range(256) if hist[i] > 0}
avg_length_huffman = sum(len(huffman_codes_mountain[pixel]) * freq[pixel] for pixel in freq) / sum(freq.values())
print(f'Gennemsnitlig kodelængde med Huffman kodning: {avg_length_huffman:.2f} bits')

# Afkod billedet og vis det
def huffman_decode(encoded_image, huffman_codes, shape):
    reverse_codes = {v: k for k, v in huffman_codes.items()}
    decoded_image = []
    buffer = ''
    for bit in encoded_image:
        buffer += bit
        if buffer in reverse_codes:
            decoded_image.append(reverse_codes[buffer])
            buffer = ''
    return np.array(decoded_image).reshape(shape)

decoded_mountain = huffman_decode(encoded_mountain, huffman_codes_mountain, image1.shape)
plt.imshow(decoded_mountain, cmap='gray')
plt.title('Afkodet Huffman billede')
plt.show()
