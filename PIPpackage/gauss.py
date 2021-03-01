# import math
import matplotlib.pyplot as plt
import numpy as np


######
# Show kernel as a 3D figure
#######################################

def plotKernel(kernel):
    arr = np.array(kernel)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')

    ax.bar3d(
        np.indices((n_points, n_points))[0].flatten(),
        np.indices((n_points, n_points))[1].flatten(),
        np.zeros_like(arr).flatten(),
        1,
        1,
        arr.flatten(),
        edgecolor='black',
        shade=False
    )
    ax.set_title('Surface plot')
    plt.show()


######
# Show kernel as a table
#######################################

def showKernel(kernel):
    for kl in kernel:
        for w in kl:
            print('%1.4f' % (w), end='\t')
        print()
    print()


######
# Calculate a single coeficiente on position r,c with sigma
#######################################

def coefi(r, c, sigma):
    return (math.exp(-0.5 * (r * r + c * c) / (sigma * sigma)))


######
# Calculate a complete Gaussian kernel of dim x dim with sigma
#######################################

def genGaussKernel(dim, sigma):
    kernel = []
    sumT = 0
    for y in range(-math.floor(dim / 2), math.floor(dim / 2) + 1):
        kline = []
        for x in range(-math.floor(dim / 2), math.floor(dim / 2) + 1):
            c = coefi(x, y, sigma)
            kline.append(c)
            sumT += c
        kernel.append(kline)
    normKernel = [[c / sumT for c in l] for l in kernel]
    return (normKernel)


######
# Separate a Gaussian kernel into a single column and line
#######################################

def separateKernel(kernel):
    line = [1]
    for i in range(1, len(kernel[0])):
        line.append(kernel[0][i] / kernel[0][0])
    return ([kernel[0], line])


######
# Convert the kernel to integer values
#######################################

def discretize(kernel):
    minV = 0
    for l in kernel:
        for c in l:
            if (minV == 0 and c > 0) or (c < minV):
                minV = c
    intKernel = kernel
    intKernel = []
    for l in kernel:
        line = [round(c / minV) for c in l]
        intKernel.append(line)
    return (intKernel)


def norm2one(kernel):
    ind = math.floor(len(kernel) / 2)
    coef = kernel[ind][ind]

    intKernel = []
    for l in kernel:
        line = [c / coef for c in l]
        intKernel.append(line)
    return (intKernel)


######
# MAIN PROGRAM
#######################################

# Parameters

n_points = 7  # to genetare a n_points by n_points kernel
sigma = 1

kernel = genGaussKernel(n_points, sigma)  # Create the kernel
showKernel(kernel)  # Show the kernel as table
sepKernel = separateKernel(kernel)  # Separate the kernel
showKernel(sepKernel)  # Show the separated kernel as table
plotKernel(kernel)
dKernel = discretize(kernel)
showKernel(dKernel)
oneKernel = norm2one(kernel)
showKernel(oneKernel)

def freqGauss(sig,freq):
    return(math.exp(-2*(math.pi*sig*freq)**2))
sig=1

xvs=[i/50 for i in range(0,51)]
yvs=[freqGauss(sig,i/100) for i in range(0,51)]
plt.plot(xvs, yvs, color='blue', linewidth=1)
# plt.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color='darkgreen', marker='^')
plt.ylim(-0.1, 1.1)
plt.show()

def calc3DBSigma(f):
    return(math.sqrt(-math.log(0.5)/(2*(math.pi*f)**2)))
import numpy

f=numpy.fft.fft2(kernel)
fs=np.fft.fftshift(f)

import blah
import os
from skimage import io
import cv2

os.chdir('/Users/josemmrf/Documents/Work new MAC/FOTOGRAFIAS/2020/2020-07-07 Oriola')
file = blah.gui_fname()
file1=str(file,encoding='utf8')

img=io.imread(file1)
img = img[:,:,2] # blue channel
plt.imshow(img, cmap='gray')

f = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
f_shift = np.fft.fftshift(f)
f_complex = f_shift[:,:,0] + 1j*f_shift[:,:,1]
f_abs = np.abs(f_complex) + 1 # lie between 1 and 1e6
f_bounded = 20 * np.log(f_abs)
f_img = 255 * f_bounded / np.max(f_bounded)
f_img = f_img.astype(np.uint8)

plt.imshow(f_img, cmap='gray')

import blah
import os
from skimage import io
import cv2 as cv
import matplotlib.pyplot as plt


####
# Calculate the DFT of imIn and show it as an image
#########

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def showDFT(imIn, tit):
    #    imgBW = rgb2gray(imIn)
    #     imgBW = imgBW-imgBW.mean()
    imgBW = imIn[:, :, 1]  # green channel

    #    f = cv.dft(imgBW, flags=cv.DFT_COMPLEX_OUTPUT)
    f = np.fft.fft2(imgBW)
    f_shift = np.fft.fftshift(f)
    f_show = np.log(1 + np.abs(f_shift))

    #    f_complex = f_shift[:,:,0] + 1j*f_shift[:,:,1]
    #    f_abs = np.abs(f_complex) + 1 # lie between 1 and 1e6
    #    f_bounded = 20 * np.log10(f_abs)
    #    f_img = 255 * f_bounded / np.max(f_bounded)
    #    f_img = f_img.astype(np.uint8)

    plt.title(tit)
    plt.ylabel('Logarithmic amplitude')
    plt.xlabel('Spacial frequencie')

    #    plt.imshow(f_img, cmap='gray')
    plt.imshow(f_show, cmap='gray')
    plt.show()

    #    return(f_img)
    return (f_show)


####
# MAIN PROGRAM
#########

os.chdir('/Users/josemmrf/Documents/Work new MAC/FOTOGRAFIAS/2020/2020-07-07 Oriola')
file = blah.gui_fname()
file1 = str(file, encoding='utf8')

img = io.imread(file1)

blur05 = cv.GaussianBlur(img, (7, 7), 0.5)
blur1 = cv.GaussianBlur(img, (7, 7), 1)
blur2 = cv.GaussianBlur(img, (13, 13), 2)
blur3 = cv.GaussianBlur(img, (19, 19), 3)
blur10 = cv.GaussianBlur(img, (61, 61), 10)

# io.imshow(img)
# io.show()
# io.imshow(blur1)
# io.show()
# io.imshow(blur3)
# io.show()
io.imshow(blur10)
io.show()

dftimg = showDFT(img, 'Original image')
dft05 = showDFT(blur05, 'After Gaussian with Sigma=0.5')
dft01 = showDFT(blur1, 'After Gaussian with Sigma=1')
dft02 = showDFT(blur2, 'After Gaussian with Sigma=2')
dft03 = showDFT(blur3, 'After Gaussian with Sigma=3')
dft10 = showDFT(blur10, 'After Gaussian with Sigma=10')

# imgZeros=np.zeros(2000,2000,3)
# blurZeros = cv.GaussianBlur(imgZeros,(61,61),10)
# dftZeros=showDFT(blurZeros,'After Gaussian with Sigma=10')

import math

def plotProfile(dft):
    (width,weight) = dft.shape

    wid=math.floor(width/2)
    wei=math.floor(weight/2)

    profile=dft[wid:width,wei]
#    plt.ylim(0, 260)
    plt.plot(profile)
    plt.show()

plotProfile(dftimg)
plotProfile(dft05)
plotProfile(dft01)
plotProfile(dft02)
plotProfile(dft03)
plotProfile(dft10)

import numpy as np
import random

imgZeros=np.zeros((1000,2000,3))
for i in range (1000):
    for j in range (2000):
        imgZeros[i,j,0]=round(random.random()*255)
        imgZeros[i,j,1]=round(random.random()*255)
        imgZeros[i,j,2]=round(random.random()*255)
imgZeros=imgZeros.astype(np.uint8)
io.imshow(imgZeros)
io.show()

blurZeros = cv.GaussianBlur(imgZeros,(51,51),7)

io.imshow(blurZeros)
io.show()

dftZeros=showDFT(blurZeros,'Zeros image after Gaussian with Sigma=10')
plotProfile(dftZeros)
dftZerosOrig=showDFT(imgZeros,'Original image')
plotProfile(dftZerosOrig)
