import numpy as np
import imageio.v3 as iio
import yuvio
import matplotlib.pyplot as plt


def Entropy(im, binsIn):
    histogram, bin_edges = np.histogram(im, bins=binsIn)
    p = histogram / np.sum(histogram)
    p1 = p[p!=0]
    entropy = -np.dot(p1.T,np.log2(p1))
    return entropy

## Exercise 5.1 Intra prediction

width = 768
height = 432
yuvformat = 'yuv420p'
videofile = 'pa_25fps.yuv'
currFrame = 55
refFrame = 50

# (a) Load frames and retrieve Y components
yuvfcurr = yuvio.imread(videofile, width, height, yuvformat, index=currFrame)
yuvfref = yuvio.imread(videofile, width, height, yuvformat, index=refFrame)
ycurrframe = yuvfcurr.y.astype(int)
yrefframe = yuvfref.y.astype(int)

figin, axin = plt.subplots(nrows=1, ncols=2)
axin[0].imshow(ycurrframe, cmap='grey')
axin[0].set_title('Frame 55')
axin[1].imshow(yrefframe, cmap='grey')
axin[1].set_title('Frame 50')
plt.show()


# (b) Prediction with predict2D
def prediction2D(signal, qf):
    signalpadded = np.pad(signal,((1, 0), (1, 0)), 'constant', constant_values=(128, 128))
    dq = np.zeros(signal.shape)
    decoded = signalpadded
    for row in range(1, signal.shape[0]):
        for col in range(1, signal.shape[1]):
            prediction = 0.33 * (decoded[row-1,col-1] + decoded[row-1,col] + decoded[row,col-1])
            d = signalpadded[row,col] - prediction

            if qf > 1:
                dq[row,col] = np.round(d/qf) * qf
            else:
                dq[row,col] = d

            decoded[row,col] = dq[row,col] + prediction

    dq = dq[1:,1:]
    decoded = decoded[1:,1:]

    return dq, decoded


# (c) Prediction for 16x16 block
frame = np.pad(ycurrframe,((1, 0), (1, 0)), 'constant', constant_values=(128, 128))
print(frame.shape)
diff = np.zeros_like(ycurrframe) # initialize diff image
blocksize = 16
# loop on blocks
for irow in range(1, ycurrframe.shape[0], blocksize):
    for icol in range(1, ycurrframe.shape[1], blocksize):
        pred =  prediction = np.mean(block)
        diff # assign difference to curent block

plt.figure()
plt.imshow(diff)
plt.colorbar()
plt.show()

print('entropy of original image is:',)
print('entropy of difference image is:',)


## Exercise 5.2 Inter prediction