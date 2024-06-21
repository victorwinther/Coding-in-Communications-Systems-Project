from PIL import Image
import numpy as np
import heapq
from collections import defaultdict, Counter

# Indlæs billedet
image = Image.open('Goldhill_bin.png')

# Konverter billedet til binært (sort/hvidt)
binary_image = image.convert('1')

# Konverter til numpy array
binary_array = np.array(binary_image)

# Uddrag billedstumpen I (5x10, startpunkt række 94, kolonne 425)
I = binary_array[94:99, 425:435]

# Udskriv billedstumpen for at tjekke værdierne
print(I)

# Run-length symboler og deres frekvenser
symbols = ['NaN', '0', '1', '2', '3']
run_length_code = ['NaN', '2', '1', 'NaN', '1', '1', '3', 'NaN', '2', '1', 'NaN', '0', '1', 'NaN', '1', '1', '1']

# Beregn frekvenser
freq = Counter(run_length_code)

# Huffman Node class
class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

# Byg Huffman træ
def build_huffman_tree(freq):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]

# Generer Huffman koder
def generate_huffman_codes(node, prefix='', codebook={}):
    if node is not None:
        if node.symbol is not None:
            codebook[node.symbol] = prefix
        generate_huffman_codes(node.left, prefix + '0', codebook)
        generate_huffman_codes(node.right, prefix + '1', codebook)
    return codebook

huffman_tree = build_huffman_tree(freq)
huffman_codes = generate_huffman_codes(huffman_tree)

# Beregn kodelængden med Huffman koder
huffman_encoded_length = sum(len(huffman_codes[symbol]) for symbol in run_length_code)

print(f"Huffman koder: {huffman_codes}")
print(f"Kodelængde med Huffman kodning: {huffman_encoded_length} bits")

huffman_stream = '011101100110110111100111011000101100110110110'

# Beregn sandsynligheden for 0
count_0 = huffman_stream.count('0')
total_bits = len(huffman_stream)
probability_0 = count_0 / total_bits

print(f"Sandsynlighed for 0: {probability_0:.4f}")

def run_length_encode(binary_array, threshold=4):
    encoded = []
    current_value = binary_array[0]
    count = 0
    
    for bit in binary_array:
        if bit == current_value:
            count += 1
        else:
            while count >= threshold:
                encoded.append('NaN')
                count -= threshold
            encoded.append(str(count))
            current_value = bit
            count = 1
    
    while count >= threshold:
        encoded.append('NaN')
        count -= threshold
    encoded.append(str(count))
    
    return encoded

# Test funktionen med billedstumpen fra opgave 2
I_flat = I.flatten()
encoded_I = run_length_encode(I_flat)
print("Run-length kodet:", encoded_I)

def run_length_decode(encoded_array, threshold=4):
    decoded = []
    current_value = 0
    
    for code in encoded_array:
        if code == 'NaN':
            decoded.extend([current_value] * threshold)
        else:
            decoded.extend([current_value] * int(code))
        current_value = 1 - current_value
    
    return decoded

# Test afkoderen
decoded_I = run_length_decode(encoded_I)
decoded_I_array = np.array(decoded_I).reshape(I.shape)
print("Afkodet:", decoded_I_array)

def run_length_encode(binary_array, threshold=4):
    encoded = []
    current_value = binary_array[0]
    count = 0
    
    for bit in binary_array:
        if bit == current_value:
            count += 1
        else:
            while count >= threshold:
                encoded.append('NaN')
                count -= threshold
            encoded.append(str(count))
            current_value = bit
            count = 1
    
    while count >= threshold:
        encoded.append('NaN')
        count -= threshold
    encoded.append(str(count))
    
    return encoded

def run_length_decode(encoded_array, threshold=4):
    decoded = []
    current_value = 0
    
    for code in encoded_array:
        if code == 'NaN':
            decoded.extend([current_value] * threshold)
        else:
            decoded.extend([current_value] * int(code))
        current_value = 1 - current_value
    
    return decoded

# Indlæs billedet
image = Image.open('Goldhill_bin.png')
binary_image = image.convert('1')
binary_array = np.array(binary_image)

# Opret billedstumpen I (5x10, startpunkt række 94, kolonne 425)
I = binary_array[94:99, 425:435]

# Run-length kodning
I_flat = I.flatten()
encoded_I = run_length_encode(I_flat)
print("Run-length kodet:", encoded_I)

# Run-length afkodning
decoded_I = run_length_decode(encoded_I)
decoded_I_array = np.array(decoded_I).reshape(I.shape)
print("Afkodet:", decoded_I_array)
