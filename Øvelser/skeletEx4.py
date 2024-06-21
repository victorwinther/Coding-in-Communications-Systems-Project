import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct, dctn, idctn
from skimage.util import apply_parallel
from Øvelser.jpeg_compression_cycle import jpeg_compression_cycle

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
plt.imshow(im_dct, cmap='gray')
plt.title('DCT Transformed Image')
plt.subplot(1, 3, 3)
plt.imshow(im_idct, cmap='gray')
plt.title('Reconstructed Image')
plt.show()
# Sæt ønskede rækker og kolonner til nul
#python
#Kopier kode
# Funktion til at sætte specifikke rækker og kolonner til nul
def setzero(block, rows, cols):
    outblock = block.copy()
    outblock[rows, :] = 0
    outblock[:, cols] = 0
    return outblock

# Anvend funktionen på billede blokke
im_dct_zeros = apply_parallel(setzero, im_dct, chunks=(8, 8), extra_arguments=(slice(4, 8), slice(4, 8)), compute=True)
im_idct_zeros = idctn(im_dct_zeros, norm='ortho')

# Visualiser billederne
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.imshow(im_dct_zeros, cmap='gray')
plt.title('DCT with zeros')
plt.subplot(1, 2, 2)
plt.imshow(im_idct_zeros, cmap='gray')
plt.title('Reconstructed from DCT with zeros')
plt.show()

print('Entropy of original image:', Entropy(im))
print('Entropy of transformed image with zeros:', Entropy(im_idct_zeros))
print('PSNR between original and transformed image with zeros:', PSNR(im, im_idct_zeros))
#d) Fjern flere coefficients
#python
#Kopier kode
# Funktion til at fjerne flere coefficients
def remove_coefficients(block, num_coeffs):
    flat = block.flatten()
    indices = np.argsort(np.abs(flat))[:-num_coeffs]
    flat[indices] = 0
    return flat.reshape(block.shape)

# Anvend funktionen på billedet
im_dct_blk_zeros = apply_parallel(remove_coefficients, im_dct, chunks=(8, 8), extra_arguments=(4,), compute=True)
im_idct_blk_zeros = idctn(im_dct_blk_zeros, norm='ortho')

# Visualiser billederne
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
ax[0, 0].imshow(im_idct_blk_zeros, cmap='gray')
ax[0, 0].set_title('DCT Blk with 4 coefficients kept')

im_dct_blk_zeros = apply_parallel(remove_coefficients, im_dct, chunks=(8, 8), extra_arguments=(1,), compute=True)
im_idct_blk_zeros = idctn(im_dct_blk_zeros, norm='ortho')
ax[0, 1].imshow(im_idct_blk_zeros, cmap='gray')
ax[0, 1].set_title('DCT Blk with 1 coefficient kept')

plt.show()

print('Entropy of original image:', Entropy(im))
print(f'Entropy of transformed image with 4 coefficients kept: {Entropy(im_idct_blk_zeros)}')
print(f'PSNR between original and image with 4 coefficients kept: {PSNR(im, im_idct_blk_zeros)}')
#e) Kvantisering med kvantiseringsmatrix
#python
#Kopier kode
# Kvantiseringsmatrix
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
QF = 10

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


# to 1D DCT operationer
#iD1 = 
#error = np.sum(array-iD1)

# 2D-DCT
#iDCT2D = 
#errDCT2D = np.sum(array-iDCT2D)
#print(f'error is: {error} for DCT1D, and {errDCT2D} for DCT2D')

#(b) 2D DCT på billede toy
im = iio.imread('../../../ImgVideo/toy.tif')
row, col = im.shape

print('entropy of original image is:',Entropy(im))
print('entropy of transformed image is:',Entropy(imdct))
print('entropy of reconstruted image is:',Entropy(imidct))
print('psnr between two images is:',PSNR(im,imidct))

#(c)
# with loops
im1c= np.zeros((row,col))
for i in range(int(row/8)):
    for j in range(int(col/8)):
        imtemp = im[i*8:i*8+8,j*8:j*8+8]
        # set desired rows & cols to 0
        im1c[i*8:i*8+8,j*8:j*8+8]=imdcttemp

im1cidct = np.zeros((row,col))
for i in range(int(row/8)):
    for j in range(int(col/8)):
        imtemp = im1c[i*8:i*8+8,j*8:j*8+8]
        # set desired rows & cols to 0
        im1cidct[i*8:i*8+8,j*8:j*8+8]=imdcttemp

plt.figure(figsize=(15, 6))
plt.subplot(1,2,1)
plt.imshow(im,cmap='gray')
plt.title('original image')
plt.subplot(1,2,2)
plt.imshow(im1cidct,cmap='gray')
plt.title('reconstructed image')

# with apply_parallel
imDCTBlk = apply_parallel(dctn, im, (8,8), extra_keywords={'norm': 'ortho'}, compute=True)

def setzero(block, idxrow, idxcol):
    outblock = block
    outblock[] = 0 # set desired rows to 0
    outblock[] = 0 # set desired colss to 0
    return outblock

# Setdesired coeff to zero using setzero
imDCTZeros = apply_parallel()

imiDCTZerBlk = apply_parallel()

plt.subplot(2,2,1)
plt.imshow(imDCTBlk,cmap='gray')
plt.title('DCT (8,8) image')
plt.subplot(2,2,2)
plt.imshow(imDCTZeros,cmap='gray')
plt.title('DCT (8,8) image blk 0')
plt.subplot(2,2,3)
plt.imshow(im,cmap='gray')
plt.title('orgn')
plt.subplot(2,2,4)
plt.imshow(imiDCTZerBlk,cmap='gray')
plt.title('Rec DCT (8,8) image blk 0')
#plt.show()

print('entropy of original image is:',Entropy(im))
print('entropy of transformed image is:',Entropy(imiDCTZerBlk))
print('psnr between two images is:',PSNR(im,imiDCTZerBlk))

#(d)
# Block process
imDCTBlkZeros4 = apply_parallel()
imiDCTZerBlk4 = apply_parallel()
imDCTBlkZeros1 = apply_parallel()
imiDCTZerBlk1 = apply_parallel()

rowzeros4 = 
colzeros4 = 
imDCTZeros4 = setzero()
imiDCTZer4 = 
rowzeros1 = 
colzeros1 = 
imDCTZeros1 = setzero()
imiDCTZer1 = 

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0, 0].imshow(imiDCTZerBlk4,cmap='gray')
ax[0, 0].set_title('DCT Blk zer4')
ax[0, 1].imshow(imiDCTZer4,cmap='gray')
ax[0, 1].set_title('DCT zer4')
ax[1, 0].imshow(imiDCTZerBlk1,cmap='gray')
ax[1, 0].set_title('DCT Blk zer1')
ax[1, 1].imshow(imiDCTZer1,cmap='gray')
ax[1, 1].set_title('DCT zer1')
#plt.show()

print('entropy of original image is:',Entropy(im))
print(f'entropy of transformed image with 4/64 coeff kept {Entropy(imiDCTZerBlk4)} for block {Entropy(imiDCTZer4)} for whole im')
print(f'psnr between two images with 4/64 coeff kept is: {PSNR(im,imiDCTZerBlk4)} for block {PSNR(im, imiDCTZer4)} for whole im')
print(f'entropy of transformed image with 1/64 coeff kept {Entropy(imiDCTZerBlk1)} for block {Entropy(imiDCTZer1)} for whole im')
print(f'psnr between two images with 1/64 coeff kept is: {PSNR(im,imiDCTZerBlk1)} for block {PSNR(im, imiDCTZer1)} for whole im')

# (f)

qm_y = np.array([[16, 11, 10, 16, 124, 140, 151, 161], [12, 12, 14, 19, 126, 158, 160, 155], 
                [14, 13, 16, 24, 140, 157, 169, 156], [14, 17, 22, 29, 151, 187, 180, 162], 
                [18, 22, 37, 56, 168, 109, 103, 177], [24, 35, 55, 64, 181, 104, 113, 192],
                [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 199]]) 

def quantizeqm(block, qm, qf):
    outblock = 
    return outblock

def unquantizeqm(block, qm, qf):
    outblock = 
    return outblock

QF = 10
imDCTBlk = apply_parallel()
imDCTBlkQM = apply_parallel()
imDCTBlkQMi = apply_parallel()
imiDCTBlkQM = np.round(apply_parallel())

print(f'entropy of original image is {Entropy(im)} of transformed with 1/64 coeff kept {Entropy(imiDCTZerBlk1)}')
print(f'entropy of transformed image with QM for QF {QF} {Entropy(imiDCTBlkQM)}')

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(imiDCTZerBlk1,cmap='gray')
ax[0].set_title('DCT Blk zer1')
ax[1].imshow(imiDCTBlkQM,cmap='gray')
ax[1].set_title('DCT BlkQM')
#plt.show()

print(f'psnr between two images is: {PSNR(im,imiDCTZerBlk1)} for 1/64 coeff kept {PSNR(im, imiDCTBlkQM)} for QM')

## Exercise 2

imParrots = iio.imread('../../../ImgVideo/parrots.bmp')
comp = jpeg_compression_cycle(imParrots)
figjpg, axjpg = plt.subplots(nrows=1, ncols=2)
axjpg[0].imshow(imParrots)
axjpg[0].set_title('org')
axjpg[1].imshow(comp)
axjpg[1].set_title('Comp')
plt.show()