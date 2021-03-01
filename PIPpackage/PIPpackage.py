from statistics import median
from math import sin, cos
from copy import deepcopy
import random
import numpy as np
from math import floor

# ADAPTED FUNCTIONS

########
# GenImage - generates an numpy array with a random image
################
def genImage(dimX,dimY,max,seed=0):
    ''' Generates a numpy array with dimX by dimY pixels of type int
    - returns the array filled with values between zero and max
    '''
    if seed!=0:
        random.seed(seed)
    img=np.empty((dimY,dimX),int)
    for x in range(dimX):
        for y in range(dimY):
            img[y,x]=random.randint(0,max)
    return(img)

########
# imagePad - padding an image an arbitrary number of lines and columns
################
def imagePad(image,padding,mode='edge'):
    ''' imagePad - padding an image
    :param image: NumPy array with the image
    :param padding: list of padding values - yUp, yDown, xLeft, xRight
    :param mode: it can be ‘edge’ (default), ‘symmetric’, 'wrap', 'constant' (padding with zeros)
    :return: NumPy array with padded image
    '''
    yu=padding[2]
    yd=padding[3]
    xl=padding[0]
    xr=padding[1]
    return(np.pad(image, ((yu, yd), (xl, xr)), mode))

########
# Print an image of integer pixels
################
def printImg(img):
    '''
    printImg - printing an INTEGER image on the screen
    :param img: NumPy array with the image to print
    :return: none
    '''
    for l in img:
        for p in l:
            print('{0:4d}'.format(p), end='')
        print()

########
# Print an image of float pixels
################
def printImgFloat(img):
    '''
    printImgFloat - printing a FLOAT image on the screen
    :param img: NumPy array with the image to print
    :return: none
    '''
    for l in img:
        for p in l:
            print('{:5.2f}'.format(p), end='')
        print()

# TO BE ADAPTED FUNCTIONS

#######
# Convolute a padded image with a kernel
################
def imgConvPad(img, kernel, verb=False):
    '''
    imgConv - Calculates the convolution of an image with a kernel
:param img: input image (NumPy array)
    :param kernel: kernel (NumPy array)
    :param verb: Verbose - True if messages are expected - default False
    :return: convolved image (NumPy array)
    '''
    (dimY, dimX) = img.shape
    (dkY, dkX) = kernel.shape
    offX=floor(dkX / 2)
    offY=floor(dkY / 2)

    imRes = np.empty([dimY-2*offY,dimX-2*offX],int)

    print(imRes.shape)
    for y in range(offY, dimY-offY):
        for x in range(offX, dimX-offX):
            s = 0
            for i in range(-offY, offY+1):
                for j in range(-offX, offX+1):
                    s += img[y + i][x + j] * kernel[i + offY][j + offX]
            imRes[y - offY][x-offX]=s

    imDiv=imRes / kernel.sum()

    if verb:
        print('Valor da soma do filtro', kernel.sum())
        print('Imagem resultado antes da divisão')
        printImg(imRes)
        print('Imagem resultado depois de dividida')
        printImgFloat(imDiv)
    return (imDiv)

########
# Convolute a unpadded image with a kernel
################
def imgConv(img, kernel):
    (dkX, dkY) = kernel.shape
    offX=floor(dkX / 2)
    offY=floor(dkY / 2)

    imgPad = imagePad(img,[offY,offY,offX,offX])
    return imgConvPad(imgPad, kernel)

########
# Filter the image img with a separated kernel filter col x lin
# Returns: floating point image with the scaled result
################
def sepFilter(col, lin, img):
    kernel=np.empty([len(col),len(lin)])
    for y in range(len(col)):
        for x in range(len(lin)):
            kernel[y][x]=col[y] * lin[x]
    print('Kernel completo (a partir do separado)')
    printImgFloat(kernel)
    return (imgConv(img, kernel))

########
# Calculate rotation source coordinates (angle in degrees)
################
def calcRotCoord(x, y, ang):
    ang = ang / 180 * pi
    xorig = x * cos(ang) - y * sin(ang)
    yorig = x * sin(ang) + y * cos(ang)
    return (xorig, yorig)

########
# Linear interpolation
################
def interp(a, b, off):
    res = a + (b - a) * off
    print('interpolating between', a, b,'at',off,'=', res)
    return (res)

########
# Bilinear interpolation
################
def bilinear(xorig, yorig, img):
    print('bilinear to find', xorig, yorig)
    x1 = floor(xorig)
    x2 = ceil(xorig)
    y1 = floor(yorig)
    y2 = ceil(yorig)
#    print(x1,x2,y1,y2,img)
    v1 = img[y1][x1]
    v2 = img[y1][x2]
    v3 = img[y2][x1]
    v4 = img[y2][x2]
    vv1 = interp(v1, v2, xorig - x1)
    vv2 = interp(v3, v4, xorig - x1)
    res = interp(vv1, vv2, yorig - y1)
    return (int(round(res,0)))

########
# Bilinear interpolation
################
def medianHibrid5x5(x, y, image,prt):
    diag = [[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2], [2, -2], [1, -1], [-1, 1], [-2, 2]]
    cross = [[0, -2], [0, -1], [0, 0], [0, 1], [0, 2], [-2, 0], [-1, 0], [1, 0], [2, 0]]

    def getValues(x, y, off, img):
        l = []
        for i in range(len(off)):
            l.append(img[y + off[i][1]][x + off[i][0]])
        return (l)

    img = imagePadDup2(image)
    if prt:
        print('Imagem com duplicacao das margens(2)\n')
        printImg(img)
        print()
    print('Pixel x=',x,'y=',y)
    l1 = getValues(x + 2, y + 2, diag, img)
    print('pixeis nas diagonais', l1)
    l2 = getValues(x + 2, y + 2, cross, img)
    print('pixeis nas verticais', l2)
    v1 = median(l1)

    v2 = median(l2)
    res = median([v1, v2, image[y][x]])
    print('mediana dos pixeis nas diagonais', v1)
    print('mediana dos pixeis nas verticais', v2)
    print('pixel central', image[y][x])
    print('resultado final da mediana hibrida', res)
    return (res)

########
# Histogram equalization
################
def eqHist(img,nLevels,verb):
    hist={}
    tot=0
    for i in range(nLevels):
        hist[i]=0
    for l in img:
        for p in l:
            hist[p]=hist[p]+1
            tot +=1
    if verb:
        print('Histogram',hist)
    accProb=hist.copy()
    prev=0
    for i in range(len(accProb)):
        prob=accProb[i]/tot
        accProb[i] = prob+prev
        prev += prob
    if verb:
        print('Accumulated probabilities: ',end='')
        for i in range(len(accProb)):
            print("{:.3f}".format(accProb[i]),end='  ')
        print()
    step=1/nLevels
    newLevels=accProb.copy()
    for i in range(len(newLevels)):
        newLevels[i]=math.ceil(newLevels[i]/step)-1
    if verb:
        print('Transformation table',newLevels)
    imgRes=deepcopy(img)
    for j in range(len(imgRes)):
        for i in range(len(imgRes[j])):
            imgRes[i][j]=newLevels[imgRes[i][j]]
    if verb:
        print('Resulting image')
        printImg(imgRes)
    return(imgRes)

########
# Sums all the values in the all picture (number of pixels 1 on binary images)
################
def numBits(img):
    tot=0
    for l in img:
        tot +=sum(l)
    return(tot)

########
# Image dilation
################
def dilation(imIn,ker):
    img=imagePadDup1(imIn)
    imRes=[]
    for y in range(1,len(img)-1):
        imRes.append([])
        for x in range(1,len(img[y])-1):
            s=0
            for i in range(-1,2):
                for j in range(-1,2):
                    if img[y+i][x+j]==1 and ker[i+1][j+1]==1:
                        s=1
            imRes[-1].append(s)
    return(imRes)

########
# Image erosion
################
def erosion(imIn,ker):
    img=imagePadDup1(imIn)
    imRes=[]
    for y in range(1,len(img)-1):
        imRes.append([])
        for x in range(1,len(img[y])-1):
            s=1
            for i in range(-1,2):
                for j in range(-1,2):
                    if img[y+i][x+j]==0 and ker[i+1][j+1]==1:
                        s=0
            imRes[-1].append(s)
    return(imRes)

########
# Hit and Miss with pad (dont cares are represented as -1 on the kernel)
################
def hitAndMiss(imIn,ker):
    img=imagePadDup1(imIn)
    imRes=[]
    for y in range(1,len(img)-1):
        imRes.append([])
        for x in range(1,len(img[y])-1):
            s=1
            for i in range(-1,2):
                for j in range(-1,2):
                    if ker[i+1][j+1]!=-1 and ker[i+1][j+1]!=img[y+i][x+j]:
                        s=0
            imRes[-1].append(s)
    return(imRes)
