# Create Huffman tree node with minimal variance for given signal
# Implementation based on exemple from geeksforgeeks.com
# Functions 
# huffmantree = buildHuffDictMV(chars, freq): Build a huffman tree with minimal variance
# decodedstring = decode_file(root, encodedString): Decode a sequence with given Huffman tree
import heapq
import numpy as np
from collections import defaultdict
from PIL import Image
import numpy as np

class NodeMV:
    def __init__(self, freq, leafdist, symbol, left=None, right=None, idxsymbol=None):
        # tuplet with frequency of symbol & distance to leaf
        self.freq_lfdist = (freq, leafdist)

        # symbol name (character)
        self.symbol = symbol

        self.idxsymbol = idxsymbol

        # node left of current node
        self.left = left

        # node right of current node
        self.right = right

        # tree direction (0/1)
        self.huff = ''

    def __lt__(self, nxt):
        return self.freq_lfdist < nxt.freq_lfdist

# utility function to print huffman
# codes for all symbols in the newly
# created Huffman tree
def printNodes(node, huffmantree, val=''):

    # huffman code for current node
    newVal = val + str(node.huff)

    # if node is not an edge node
    # then traverse inside it
    if(node.left):
        printNodes(node.left, huffmantree, newVal)
    if(node.right):
        printNodes(node.right, huffmantree, newVal)

        # if node is edge node then
        # display its huffman code
    if(not node.left and not node.right):
        print(f"{node.symbol} -> {newVal}")
        huffmantree[node.idxsymbol] = (node.symbol, newVal)
        #huffmantree[node.idxsymbol, 1] = newVal

    return huffmantree

def buildHuffDictMV(chars, freq):
    # Build a huffman tree with minimal variance
    # Input: chars - list of symbols, freq - list of probabilities of occurence
    # Output: huffmantree - numpy structured array 1st column 'symbol', 2nd column 'code'
    # NB: max size of code is 32 bits

    # list containing unused nodes
    nodes = []
    orgnleafdist = 1
    # converting characters and frequencies
    # into huffman tree nodes
    for x in range(len(chars)):
        heapq.heappush(nodes, NodeMV(freq[x], orgnleafdist, chars[x], idxsymbol=x))

    while len(nodes) > 1:

        # sort all the nodes in ascending order
        # based on their frequency
        left = heapq.heappop(nodes)
        right = heapq.heappop(nodes)

        # assign directional value to these nodes
        left.huff = 0
        right.huff = 1

        # combine the 2 smallest nodes to create
        # new node as their parent
        newnode = NodeMV(left.freq_lfdist[0]+right.freq_lfdist[0], min(left.freq_lfdist[1], right.freq_lfdist[1])+1, left.symbol+right.symbol, left, right)

        heapq.heappush(nodes, newnode)

    # Huffman Tree is ready!
    huffmantable = np.zeros(len(freq), dtype=[('symbol', np.dtype('U1')), ('code', np.dtype('U32'))]) #np.dtype(np.int16)
    huffmantable = printNodes(nodes[0], huffmantable)
    huffmantree = nodes[0]
    return huffmantable, huffmantree

def decodehuff(huffmannode, encodedString):
# Decode the input signal with given Huffman tree
# Inputs: encodedString - sequence to decode, huffmannode - Huffman tree
# Output: decodedarray - decoded sequence
# function iterates through the encoded signal
# if s[i]=='1' then move to node->right
# if s[i]=='0' then move to node->left
# if leaf node append the node->symbol to our output array
    ans = np.empty((0,),dtype=float)
    curr = huffmannode
    n = len(encodedString)
    for i in range(n):
        if encodedString[i] == '0':
            curr = curr.left
        else:
            curr = curr.right
 
        # reached leaf node
        if curr.left is None and curr.right is None:
            ans = np.append(ans, float(curr.symbol))
            curr = huffmannode

    decodedarray = ans
    return decodedarray

# Eksempel på brug:
chars = ['a', 'b', 'c', 'd', 'e']
freq = [0.39, 0.08, 0.21, 0.13, 0.19]

huffmantable, huffmantree = buildHuffDictMV(chars, freq)

# Udskriv Huffman koder
print("Huffman Codes:")
for symbol, code in huffmantable:
    print(f"{symbol}: {code}")

# Beregn den gennemsnitlige kodelængde
avg_length = sum([freq[i] * len(huffmantable[i][1]) for i in range(len(chars))])
print(f"Gennemsnitlig kodelængde: {avg_length:.2f} bits")