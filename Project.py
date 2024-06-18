import numpy as np
import struct
from collections import defaultdict

def read_bytes(data, num_bytes):
    return data[:num_bytes], data[num_bytes:]

def read_byte(data):
    return data[0], data[1:]

def read_word(data):
    return (data[0] << 8) + data[1], data[2:]

def zigzag_order():
    return [
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63
    ]

def build_huffman_table(bits, values):
    hufftable = {}
    code = 0
    length = 1
    for bit_count, symbols in zip(bits, values):
        for symbol in symbols:
            hufftable[symbol] = (code, length)
            code += 1
        code <<= 1
        length += 1
    return hufftable

class JPEGDecoder:
    def __init__(self, data):
        self.data = data
        self.huffman_tables = {}
        self.quantization_tables = {}
        self.dc_tables = {}
        self.ac_tables = {}
        self.width = 0
        self.height = 0
        self.components = []
        self.restart_interval = 0
        self.sof0 = None

    def decode(self):
        data = self.data
        while data:
            marker, data = read_word(data)
            if marker == 0xFFD8:  # SOI
                continue
            elif marker == 0xFFD9:  # EOI
                break
            elif marker == 0xFFDB:  # DQT
                data = self.decode_dqt(data)
            elif marker == 0xFFC4:  # DHT
                data = self.decode_dht(data)
            elif marker == 0xFFDA:  # SOS
                data = self.decode_sos(data)
            elif marker == 0xFFC0:  # SOF0
                data = self.decode_sof0(data)
            elif marker == 0xFFDD:  # DRI
                data = self.decode_dri(data)
            else:
                length, data = read_word(data)
                data = data[length - 2:]

    def decode_dqt(self, data):
        length, data = read_word(data)
        length -= 2
        while length > 0:
            qt_info, data = read_byte(data)
            qt_id = qt_info & 0x0F
            qt_precision = qt_info >> 4
            qt_size = 64 * (qt_precision + 1)
            qt_data, data = read_bytes(data, qt_size)
            qt_table = np.array(struct.unpack('B' * qt_size, qt_data)).reshape((8, 8))
            self.quantization_tables[qt_id] = qt_table
            length -= 1 + qt_size
        return data

    def decode_dht(self, data):
        length, data = read_word(data)
        length -= 2
        while length > 0:
            ht_info, data = read_byte(data)
            ht_type = ht_info >> 4
            ht_id = ht_info & 0x0F
            bits, data = read_bytes(data, 16)
            bits = list(bits)
            values = []
            for bit_count in bits:
                value, data = read_bytes(data, bit_count)
                values.append(list(value))
            hufftable = build_huffman_table(bits, values)
            if ht_type == 0:
                self.dc_tables[ht_id] = hufftable
            else:
                self.ac_tables[ht_id] = hufftable
            length -= 1 + 16 + sum(bits)
        return data

    def decode_sof0(self, data):
        length, data = read_word(data)
        precision, data = read_byte(data)
        self.height, data = read_word(data)
        self.width, data = read_word(data)
        num_components, data = read_byte(data)
        self.components = []
        for _ in range(num_components):
            component_id, data = read_byte(data)
            sampling_factors, data = read_byte(data)
            qt_id, data = read_byte(data)
            self.components.append({
                'id': component_id,
                'h': sampling_factors >> 4,
                'v': sampling_factors & 0x0F,
                'qt': self.quantization_tables[qt_id]
            })
        return data

    def decode_sos(self, data):
        length, data = read_word(data)
        num_components, data = read_byte(data)
        for _ in range(num_components):
            component_id, data = read_byte(data)
            ht_ids, data = read_byte(data)
            dc_table = self.dc_tables[ht_ids >> 4]
            ac_table = self.ac_tables[ht_ids & 0x0F]
            self.components[component_id - 1]['dc_table'] = dc_table
            self.components[component_id - 1]['ac_table'] = ac_table
        ss, data = read_byte(data)
        se, data = read_byte(data)
        ah_al, data = read_byte(data)
        length -= 2 + 2 * num_components
        scan_data, data = read_bytes(data, length)
        # Handle scan data here
        return data

    def decode_dri(self, data):
        length, data = read_word(data)
        self.restart_interval, data = read_word(data)
        return data

    # Additional methods for Huffman decoding, IDCT, dequantization, etc. go here.

def huffman_decode(bitstream, table):
    value = 0
    length = 0
    while True:
        value = (value << 1) | bitstream.read(1)
        length += 1
        for symbol, (code, code_length) in table.items():
            if length == code_length and value == code:
                return symbol
        if length > 16:
            raise ValueError("Huffman code not found")

def dequantize(block, qt):
    return block * qt

def idct_1d(vector):
    N = len(vector)
    result = np.zeros_like(vector, dtype=float)
    for k in range(N):
        sum = 0
        for n in range(N):
            sum += vector[n] * np.cos((np.pi / N) * (n + 0.5) * k)
        result[k] = sum
    return result

def idct_2d(block):
    return np.array([idct_1d(row) for row in idct_1d(block.T)]).T

class BitStream:
    def __init__(self, data):
        self.data = data
        self.byte_index = 0
        self.bit_index = 0

    def read(self, num_bits):
        value = 0
        for _ in range(num_bits):
            if self.bit_index == 0:
                self.current_byte = self.data[self.byte_index]
                self.byte_index += 1
            value = (value << 1) | ((self.current_byte >> (7 - self.bit_index)) & 1)
            self.bit_index = (self.bit_index + 1) % 8
        return value
    
def main():
    with open("example.jpg", "rb") as f:
        data = f.read()

    decoder = JPEGDecoder(data)
    decoder.decode()
    # Process and display the decoded image data

if __name__ == "__main__":
    main()

