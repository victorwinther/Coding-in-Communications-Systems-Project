import numpy as np
from PIL import Image
import cv2
from heapq import heappop, heappush
from collections import Counter

def rgb_to_ycbcr(image):
    ycbcr = np.zeros_like(image, dtype=np.float32)
    ycbcr[..., 0] =  0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    ycbcr[..., 1] = -0.1687 * image[..., 0] - 0.3313 * image[..., 1] + 0.5 * image[..., 2] + 128
    ycbcr[..., 2] =  0.5 * image[..., 0] - 0.4187 * image[..., 1] - 0.0813 * image[..., 2] + 128
    return ycbcr

def downsample(channel):
    h, w = channel.shape
    downsampled = channel.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    return downsampled

def dct_2d(block):
    return np.round(cv2.dct(block))

def block_process(channel, block_size, process_block):
    print(f"Processing blocks of shape: {channel.shape}")
    if len(channel.shape) == 3:
        channel = channel.reshape(-1, channel.shape[-2], channel.shape[-1])
    h, w = channel.shape[:2]
    blocks = (channel.reshape(h // block_size, block_size, -1, block_size)
                      .swapaxes(1, 2)
                      .reshape(-1, block_size, block_size))
    processed_blocks = np.array([process_block(block) for block in blocks])
    return (processed_blocks.reshape(h // block_size, w // block_size, block_size, block_size)
                            .swapaxes(1, 2)
                            .reshape(h, w))

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

def quantize(block, quant_table):
    return np.round(block / quant_table).astype(np.int32)

def dequantize(block, quant_table):
    return block * quant_table

def build_huffman_tree(symbols):
    heap = [[weight, [symbol, ""]] for symbol, weight in Counter(symbols).items()]
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return dict(sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))

def huffman_encode(data, huff_table):
    return "".join(huff_table[symbol] for symbol in data)

def huffman_decode(encoded_data, huff_table):
    reverse_huff_table = {v: k for k, v in huff_table.items()}
    decoded_data = []
    buffer = ""
    for bit in encoded_data:
        buffer += bit
        if buffer in reverse_huff_table:
            decoded_data.append(reverse_huff_table[buffer])
            buffer = ""
    return np.array(decoded_data)

def zigzag_order(block):
    indexorder = sorted(((x, y) for x in range(8) for y in range(8)), key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1]))
    return [block[i, j] for i, j in indexorder]

def inverse_zigzag_order(data):
    indexorder = sorted(((x, y) for x in range(8) for y in range(8)), key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1]))
    block = np.zeros((8, 8), dtype=np.int32)
    for i, (x, y) in enumerate(indexorder):
        block[x, y] = data[i]
    return block

def encode_channel(channel_quantized):
    huff_table = build_huffman_tree(channel_quantized.flatten())
    encoded_data = huffman_encode(zigzag_order(channel_quantized), huff_table)
    return encoded_data, huff_table

def decode_channel(encoded_data, huff_table):
    decoded_data = huffman_decode(encoded_data, huff_table)
    decoded_blocks = np.array([inverse_zigzag_order(decoded_data[i*64:(i+1)*64]) for i in range(len(decoded_data) // 64)])
    return decoded_blocks

def idct_2d(block):
    return cv2.idct(block.astype(np.float32))

def upsample(channel, target_shape):
    return cv2.resize(channel, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

def ycbcr_to_rgb(ycbcr):
    rgb = np.zeros_like(ycbcr, dtype=np.uint8)
    rgb[..., 0] = ycbcr[..., 0] + 1.402 * (ycbcr[..., 2] - 128)
    rgb[..., 1] = ycbcr[..., 0] - 0.344136 * (ycbcr[..., 1] - 128) - 0.714136 * (ycbcr[..., 2] - 128)
    rgb[..., 2] = ycbcr[..., 0] + 1.772 * (ycbcr[..., 1] - 128)
    return np.clip(rgb, 0, 255).astype(np.uint8)

# Load and process the image
image = np.array(Image.open('profile2.jpg'))
ycbcr_image = rgb_to_ycbcr(image)
y = ycbcr_image[..., 0]
cb = downsample(ycbcr_image[..., 1])
cr = downsample(ycbcr_image[..., 2])
y_dct = block_process(y, 8, dct_2d)
cb_dct = block_process(cb, 8, dct_2d)
cr_dct = block_process(cr, 8, dct_2d)
y_quantized = block_process(y_dct, 8, lambda block: quantize(block, quant_table_luminance))
cb_quantized = block_process(cb_dct, 8, lambda block: quantize(block, quant_table_chrominance))
cr_quantized = block_process(cr_dct, 8, lambda block: quantize(block, quant_table_chrominance))
y_encoded, y_huff_table = encode_channel(y_quantized)
cb_encoded, cb_huff_table = encode_channel(cb_quantized)
cr_encoded, cr_huff_table = encode_channel(cr_quantized)

# Decode
y_decoded_blocks = decode_channel(y_encoded, y_huff_table)
cb_decoded_blocks = decode_channel(cb_encoded, cb_huff_table)
cr_decoded_blocks = decode_channel(cr_encoded, cr_huff_table)
print(f"Y decoded blocks shape: {y_decoded_blocks.shape}")
print(f"Cb decoded blocks shape: {cb_decoded_blocks.shape}")
print(f"Cr decoded blocks shape: {cr_decoded_blocks.shape}")
y_dequantized = block_process(y_decoded_blocks, 8, lambda block: dequantize(block, quant_table_luminance))
cb_dequantized = block_process(cb_decoded_blocks, 8, lambda block: dequantize(block, quant_table_chrominance))
cr_dequantized = block_process(cr_decoded_blocks, 8, lambda block: dequantize(block, quant_table_chrominance))
y_idct = block_process(y_dequantized, 8, idct_2d)
cb_idct = block_process(cb_dequantized, 8, idct_2d)
cr_idct = block_process(cr_dequantized, 8, idct_2d)
cb_upsampled = upsample(cb_idct, y.shape)
cr_upsampled = upsample(cr_idct, y.shape)
ycbcr_decoded = np.stack((y_idct, cb_upsampled, cr_upsampled), axis=-1)
rgb_decoded = ycbcr_to_rgb(ycbcr_decoded)

# Show the resulting image
decoded_image = Image.fromarray(rgb_decoded)
decoded_image.show()
