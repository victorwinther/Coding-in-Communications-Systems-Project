import numpy as np
import matplotlib.pyplot as plt
import yuvio
from scipy.fftpack import dctn
from skimage.util import apply_parallel
# Video og frame information
width = 768
height = 432
yuvformat = 'yuv420p'
videofile = 'pa_25fps.yuv'
currFrame = 55
refFrame = 50

# Indlæs frames og hent luminanskomponenten (Y)
yuvfcurr = yuvio.imread(videofile, width, height, yuvformat, index=currFrame)
yuvfref = yuvio.imread(videofile, width, height, yuvformat, index=refFrame)
ycurrframe = yuvfcurr.y.astype(int)
yrefframe = yuvfref.y.astype(int)

# Vis luminansværdierne som billeder
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
ax[0].imshow(ycurrframe, cmap='gray')
ax[0].set_title('Frame 55')
ax[1].imshow(yrefframe, cmap='gray')
ax[1].set_title('Frame 50')
plt.show()

# Pixel-baseret prædiktion (fra øvelse 3.3)
def prediction2D(signal, qf=1):
    signalpadded = np.pad(signal, ((1, 0), (1, 0)), 'constant', constant_values=(128, 128))
    dq = np.zeros(signal.shape)
    decoded = signalpadded
    for row in range(1, signal.shape[0] + 1):
        for col in range(1, signal.shape[1] + 1):
            prediction = 0.33 * (decoded[row - 1, col - 1] + decoded[row - 1, col] + decoded[row, col - 1])
            d = signalpadded[row, col] - prediction

            if qf > 1:
                dq[row - 1, col - 1] = np.round(d / qf) * qf
            else:
                dq[row - 1, col - 1] = d

            decoded[row, col] = dq[row - 1, col - 1] + prediction

    return dq, decoded[1:, 1:]

# Beregn prædiktionsfejl for hver frame
diff_55, _ = prediction2D(ycurrframe)
diff_50, _ = prediction2D(yrefframe)

# Vis forskellen som billede
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
ax[0].imshow(diff_55, cmap='gray', vmin=-128, vmax=128)
ax[0].set_title('Prediction Error Frame 55')
ax[1].imshow(diff_50, cmap='gray', vmin=-128, vmax=128)
ax[1].set_title('Prediction Error Frame 50')
plt.show()

# Beregn entropi
def Entropy(signal):
    unique, counts = np.unique(signal, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

print('Entropy of Frame 55 Prediction Error:', Entropy(diff_55))
print('Entropy of Frame 50 Prediction Error:', Entropy(diff_50))

# 16x16 blokbaseret prædiktion
blocksize = 16
frame = np.pad(ycurrframe, ((1, 0), (1, 0)), 'constant', constant_values=(128, 128))
diff_blocks = np.zeros_like(ycurrframe)


for i in range(0, ycurrframe.shape[0], blocksize):
    for j in range(0, ycurrframe.shape[1], blocksize):
        block = frame[i:i+blocksize+1, j:j+blocksize+1]  # Hent 17x17 blok
        prediction = np.mean(block)
        diff_blocks[i:i+blocksize, j:j+blocksize] = block[1:, 1:] - prediction

plt.imshow(diff_blocks, cmap='gray', vmin=-128, vmax=128)
plt.title('16x16 Block Prediction Error Frame 55')
plt.colorbar()
plt.show()

print('Entropy of original image:', Entropy(ycurrframe))
print('Entropy of 16x16 block difference image:', Entropy(diff_blocks))


# 16x16 blokbaseret 2D DCT
def block_dct(frame, blocksize=16):
    return apply_parallel(dctn, frame, chunks=(blocksize, blocksize), extra_keywords={'norm': 'ortho'}, compute=True)

dct_50 = block_dct(ycurrframe)
dct_55 = block_dct(yrefframe)

# Histogram og antal koefficienter med absolut værdi over 20
hist_50, _ = np.histogram(dct_50, bins=1000)
hist_55, _ = np.histogram(dct_55, bins=1000)

num_coeff_50 = np.sum(np.abs(dct_50) > 20)
num_coeff_55 = np.sum(np.abs(dct_55) > 20)

plt.figure()
plt.hist(dct_50.flatten(), bins=1000, alpha=0.5, label='Frame 50 DCT')
plt.hist(dct_55.flatten(), bins=1000, alpha=0.5, label='Frame 55 DCT')
plt.legend()
plt.title('DCT Coefficient Histogram')
plt.show()

print('Number of coefficients > 20 in Frame 50:', num_coeff_50)
print('Number of coefficients > 20 in Frame 55:', num_coeff_55)

# Beregn forskellen mellem de to frames
diff_frame = ycurrframe - yrefframe

# Udfør 16x16 blokbaseret 2D DCT på residualerne
dct_diff = block_dct(diff_frame)

# Histogram og antal koefficienter med absolut værdi over 20 for residualerne
hist_diff, _ = np.histogram(dct_diff, bins=1000)
num_coeff_diff = np.sum(np.abs(dct_diff) > 20)

plt.figure()
plt.hist(dct_diff.flatten(), bins=1000, alpha=0.5, label='Difference DCT')
plt.legend()
plt.title('DCT Coefficient Histogram of Difference')
plt.show()

print('Number of coefficients > 20 in Difference Frame:', num_coeff_diff)

print('Number of coefficients > 20 in Frame 50:', num_coeff_50)
print('Number of coefficients > 20 in Frame 55:', num_coeff_55)
print('Number of coefficients > 20 in Difference Frame:', num_coeff_diff)

# Analyse
if num_coeff_diff < min(num_coeff_50, num_coeff_55):
    print("Inter prediction (difference DCT) is better for compression.")
else:
    print("Intra prediction (individual frame DCT) is better for compression.")
