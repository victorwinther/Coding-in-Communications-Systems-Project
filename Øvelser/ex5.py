import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct, dctn, idctn
from skimage.util import apply_parallel
from jpeg_compression_cycle import jpeg_compression_cycle

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

## Exercise 1
#(a) Sammenlign DCT-1D og DCT-2D
array = np.random.randint(0, 256, size=(8, 8), dtype=np.uint8)


# Udfør 2D DCT ved hjælp af to 1D DCT operationer
dct_1d_horiz = dct(array, axis=0, norm='ortho')
dct_1d_vert = dct(dct_1d_horiz, axis=1, norm='ortho')

# Udfør direkte 2D DCT
dct_2d = dctn(array, norm='ortho')

# Sammenlign resultaterne
error = np.sum(np.abs(dct_2d - dct_1d_vert))
print(f'Error between 1D and 2D DCT: {error}')

# Visualiser resultatet
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(array, cmap='gray')
plt.title('Original 8x8 Block')
plt.subplot(1, 3, 2)
plt.imshow(dct_1d_vert, cmap='gray')
plt.title('DCT using 1D operations')
plt.subplot(1, 3, 3)
plt.imshow(dct_2d, cmap='gray')
plt.title('Direct 2D DCT')
plt.show()

# Indlæs billede
im = iio.imread('pictures/boat.tif')
row, col = im.shape

# Udfør 2D DCT på hele billedet
im_dct = dctn(im, norm='ortho')
im_idct = idctn(im_dct, norm='ortho')

print('Entropy of original image:', Entropy(im))
print('Entropy of transformed image:', Entropy(im_dct))
print('Entropy of reconstructed image:', Entropy(im_idct))
print('PSNR between original and reconstructed image:', PSNR(im, im_idct))

# Visualiser billederne
plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1)
plt.imshow(im, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(np.log1p(np.abs(im_dct)), cmap='gray')
plt.title('DCT Transformed Image')
plt.subplot(1, 3, 3)
plt.imshow(im_idct, cmap='gray')
plt.title('Reconstructed Image')
plt.show()

# Funktion til at sætte 75% af koefficienterne til 0
def set_zero_coefficients(block, num_coeff):
    flat = block.flatten()
    indices = np.argsort(np.abs(flat))
    flat[indices[:-num_coeff]] = 0
    return flat.reshape(block.shape)

# Anvend funktionen på billede blokke (kun 4 koefficienter beholdt)
im_dct_zeros_4 = apply_parallel(set_zero_coefficients, im_dct, chunks=(8, 8), extra_arguments=(4,), compute=True)
im_idct_zeros_4 = idctn(im_dct_zeros_4, norm='ortho')

# Anvend funktionen på billede blokke (kun 1 koefficient beholdt)
im_dct_zeros_1 = apply_parallel(set_zero_coefficients, im_dct, chunks=(8, 8), extra_arguments=(1,), compute=True)
im_idct_zeros_1 = idctn(im_dct_zeros_1, norm='ortho')

# Visualiser billederne
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.imshow(im_idct_zeros_4, cmap='gray')
plt.title('Reconstructed from DCT with 4 coefficients kept')
plt.subplot(1, 2, 2)
plt.imshow(im_idct_zeros_1, cmap='gray')
plt.title('Reconstructed from DCT with 1 coefficient kept')
plt.show()

print('Entropy of original image:', Entropy(im))
print('Entropy of reconstructed image with 4 coefficients:', Entropy(im_idct_zeros_4))
print('PSNR between original and reconstructed image with 4 coefficients:', PSNR(im, im_idct_zeros_4))
print('Entropy of reconstructed image with 1 coefficient:', Entropy(im_idct_zeros_1))
print('PSNR between original and reconstructed image with 1 coefficient:', PSNR(im, im_idct_zeros_1))

qm_y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], 
    [12, 12, 14, 19, 26, 58, 60, 55], 
    [14, 13, 16, 24, 40, 57, 69, 56], 
    [14, 17, 22, 29, 51, 87, 80, 62], 
    [18, 22, 37, 56, 68, 109, 103, 77], 
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], 
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def quantizeqm(block, qm, qf):
    return np.round(block / (qm * qf))

def unquantizeqm(block, qm, qf):
    return block * (qm * qf)

# Kvantiseringsfaktor
QF = 12

# Kvantisér og dekod billedet
im_dct_blk_qm = apply_parallel(quantizeqm, im_dct, chunks=(8, 8), extra_arguments=(qm_y, QF), compute=True)
im_dct_blk_qm_un = apply_parallel(unquantizeqm, im_dct_blk_qm, chunks=(8, 8), extra_arguments=(qm_y, QF), compute=True)
im_idct_blk_qm = idctn(im_dct_blk_qm_un, norm='ortho')

# Visualiser billederne
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
ax[0].imshow(im_idct_blk_qm, cmap='gray')
ax[0].set_title('DCT with Quantization Matrix')
plt.show()

print(f'PSNR between original and QM image: {PSNR(im, im_idct_blk_qm)}')
#Denne kode udfører de forskellige trin i øvelse

