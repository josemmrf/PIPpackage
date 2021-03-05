from statistics import median
from math import sin, cos
from copy import deepcopy
import random
import numpy as np
from math import floor, ceil

# ADAPTED FUNCTIONS

########
# GenImage - generates an numpy array with a random image
################
def genImage(dimX,dimY,max,seed=0):
    '''
    Generates a numpy array with dimX by dimY pixels of type int
    - returns the array filled with values between zero and max
    '''
    if seed!=0:
        random.seed(seed)
    img=np.empty((dimY,dimX),int)
    for x in range(dimX):
        for y in range(dimY):
            img[y,x]=random.randint(0,max)
    return img

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
    return np.pad(image, ((yu, yd), (xl, xr)), mode)

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
    Calculates the convolution of an image with a kernel. The image must come with padding already
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

#    print(imRes.shape)
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
    return imDiv

########
# Convolute a unpadded image with a kernel
################
def imgConv(img, kernel):
    '''
    Does the padding (edge mode) and convolutes the image with a kernel
    :param img: input image without padding
    :param kernel: kernel to be used (any dimension)
    :return: convoluted image
    '''
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
    '''
    Performs a convolutional filter specified on separated form
    :param col: filter column (any ODD dimension)
    :param lin: filter line (any ODD dimension)
    :param img: image to be filtered
    :return: filtered image
    '''
    kernel=np.empty([len(col),len(lin)])
    for y in range(len(col)):
        for x in range(len(lin)):
            kernel[y][x]=col[y] * lin[x]
#    print('Kernel completo (a partir do separado)')
#    printImgFloat(kernel)
    return imgConv(img, kernel)

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
def interp(a, b, off, verb=False):
    '''
    Linear interpolation
    :param a: First value
    :param b: Second alue
    :param off: Offset between first and second value
    :param verb: True if messages are expected
    :return: exact interpolation value (not rounded)
    '''
    res = a + (b - a) * off
    if verb:
        print('Interpolating between', a,'and', b,'at',off,'=', res)
    return res

########
# Bilinear interpolation
################
def bilinear(xorig, yorig, img, verb=False):
    '''
    Bilinear interpolation
    :param xorig: X coordinate
    :param yorig: Y coordinate
    :param img: Image to interpolate
    :param verb: True if messages are expected
    :return: rounded value of resulting bilinear interpolation
    '''
    if verb:
        print('Bilinear to find', xorig, yorig)
    x1 = floor(xorig)
    x2 = ceil(xorig)
    y1 = floor(yorig)
    y2 = ceil(yorig)
    v1 = img[y1][x1]
    v2 = img[y1][x2]
    v3 = img[y2][x1]
    v4 = img[y2][x2]
    if verb:
        print('Values to consider:',v1,v2,v3,v4)
    vv1 = interp(v1, v2, xorig - x1,verb)
    vv2 = interp(v3, v4, xorig - x1,verb)
    res = interp(vv1, vv2, yorig - y1,verb)
    return int(round(res,0))

########
# Hybrid 5x5 median filter value for a pixel
################
def medianHibrid5x5(x, y, image,verb=False):
    '''
    Hybrid median filter
    :param x: X coordinate
    :param y: Y coordinate
    :param image: image to be filtered
    :param prt:
    :return:
    '''
    diag = [[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2], [2, -2], [1, -1], [-1, 1], [-2, 2]]
    cross = [[0, -2], [0, -1], [0, 0], [0, 1], [0, 2], [-2, 0], [-1, 0], [1, 0], [2, 0]]

    def getValues(x, y, off, img):
        l = np.empty([len(off)])
        for i in range(len(off)):
            l[i]=img[y + off[i][1]][x + off[i][0]]
        return (l)

    img = imagePad(image,[2,2,2,2])
    if verb:
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
    return res

########
# Histogram equalization
################
def eqHist(img,maxLevel,verb=False):
    '''
    Histogram equalization
    :param img: image to equalize
    :param maxLevel: maximum level of gray
    :param verb: True if messages are expected
    :return: equalized image
    '''
    hist={v:0 for v in range(maxLevel+1)}  # Initialize histogram with zero for all the levels
    tot=0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            hist[round(img[y][x],0)] = hist[round(img[y][x],0)]+1
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
    step=1/(maxLevel+1)
    newLevels=accProb.copy()
    for i in range(len(newLevels)):
        newLevels[i]=ceil(newLevels[i]/step)-1
    if verb:
        print('Transformation table',newLevels)
    imgRes=deepcopy(img)
    for j in range(len(imgRes)):
        for i in range(len(imgRes[j])):
            imgRes[i][j]=newLevels[imgRes[i][j]]
    if verb:
        print('Resulting image')
        printImg(imgRes)
    return imgRes

########
# Sums all the values in the all picture (number of pixels 1 on binary images)
################
def numBits(img):
    '''
    Number of bits
    :param img: input BINARY image
    :return: number of pixels = 1 in the image
    '''
    return img.sum()

########
# Image dilation
################
def dilation(imIn,ker):
    '''
    Dilation of a binary image
    :param imIn: binary image to dilate
    :param ker: dilation kernel
    :return: dilated image
    '''
    (dkY, dkX) = ker.shape
    offX=floor(dkX / 2)
    offY=floor(dkY / 2)
    img = imagePad(imIn,[offY,offY,offX,offX])
    (dimY, dimX) = img.shape
    imRes=np.empty_like(imIn)

    for y in range(offY,dimY-offY):
        for x in range(offX,dimX-offX):
            imRes[y-offY][x-offX]=0
            for i in range(-offY,offY+1):
                for j in range(-offX,offX+1):
                    if img[y+i][x+j]==1 and ker[i+1][j+1]==1:
                        imRes[y-offY][x-offX]=1
    return imRes

########
# Image erosion
################
def erosion(imIn,ker):
    '''
    Erodes a binary image with the binary kernel ker
    :param imIn:
    :param ker:
    :return:
    '''
    (dkY, dkX) = ker.shape
    offX=floor(dkX / 2)
    offY=floor(dkY / 2)
    img = imagePad(imIn,[offY,offY,offX,offX])
    (dimY, dimX) = img.shape
    imRes=np.empty_like(imIn)

    for y in range(offY,dimY-offY):
        for x in range(offX,dimX-offX):
            imRes[y-offY][x-offX]=1
            for i in range(-offY,offY+1):
                for j in range(-offX,offX+1):
                    if img[y+i][x+j]==0 and ker[i+1][j+1]==1:
                        imRes[y-offY][x-offX]=0
    return imRes

########
# Hit and Miss with pad (dont cares are represented as -1 on the kernel)
################
def hitAndMiss(imIn,ker):
    '''
    Hit and Miss operation
    :param imIn: input image
    :param ker: kernel
    :return: processed image
    '''

    (dkY, dkX) = ker.shape
    offX=floor(dkX / 2)
    offY=floor(dkY / 2)
    img = imagePad(imIn,[offY,offY,offX,offX])
    (dimY, dimX) = img.shape
    imRes=np.empty_like(imIn)

    for y in range(offY,dimY-offY):
        for x in range(offX,dimX-offX):
            imRes[y-offY][x-offX]=1
            for i in range(-offY,offY+1):
                for j in range(-offX,offX+1):
                    if ker[i+1][j+1]!=-1 and ker[i+1][j+1]!=img[y+i][x+j]:
                        imRes[y-offY][x-offX]=0
    return imRes

def multiSeg(img, verb=False):
    '''
    Segmentation of multiclass images
    :param img: input image with any number of labels
    :param verb: True if messages are expected
    :return: labelled image with a different label for each object and the number of labels on the resulting image
             The result will have labels from 0 sequentially to the number of labels-1
    '''

    def insertConf(cTab, l1, l2):
        for i in range(len(cTab)):
            if l1 in cTab[i] and l2 in cTab[i]:
                return (cTab)  # Label already registered
            elif l1 in cTab[i]:
                cTab[i].append(l2)
                return (cTab)
            elif l2 in cTab[i]:
                cTab[i].append(l1)
                return (cTab)
        cTab.append([l1, l2])
        return (cTab)

    def labEq(lab, confTab):
        for i in range(len(confTab)):
            if lab in confTab[i]:
                return (min(confTab[i]))
        return (lab)

    (height,width)=img.shape
    labels=np.empty_like(img,int)

    # Initial pass - place first lables

    nextLabel=0
    labels[0][0]=nextLabel   # First pixel
    nextLabel +=1
    for x in range(1,width):     # First row
        if img[0][x]==img[0][x-1]:
            labels[0][x]=labels[0][x-1]
        else:
            labels[0][x]=nextLabel
            nextLabel += 1

    confTab=[]

    for y in range(1,height):    # Remaining rows
        if img[y][0]==img[y-1][0]:  # First column
            labels[y][0]=labels[y-1][0]
        else:
            labels[y][0]=nextLabel
            nextLabel += 1
        for x in range(1,width):        # Remaining columns
            if img[y][x]==img[y-1][x]:
                labels[y][x]=labels[y-1][x]
                if img[y][x]==img[y][x-1] and labels[y][x]!=labels[y][x-1]:
                    confTab=insertConf(confTab,labels[y][x-1],labels[y][x])
            elif img[y][x]==img[y][x-1]:
                labels[y][x]=labels[y][x-1]
            else:
                labels[y][x]=nextLabel
                nextLabel += 1
    labDict={}
    off=0
    for l in range(nextLabel):
        le=labEq(l,confTab)
        if l!=le:
            labDict[l]=labDict[le]
            off +=1
        else:
            labDict[l]=l-off
    if verb:
        print('Input image')
        print(img)
        print('Labels before equivalencies')
        print(labels)
        print('Conflit table')
        print(confTab)
        print('Number of labels before equivalencies:',nextLabel)
        print('Dictionary')
        print(labDict)
        print('Number of labels after equivalencies:',nextLabel-off)

    for y in range(0,height):
        for x in range(0,width):
            labels[y][x]=labDict[labels[y][x]]

    if verb:
        print('Labels after equivalencies')
        print(labels)

    return(labels,nextLabel-off)

def binSeg(img, verb=False):
    '''
    Binaru segmentation
    :param img: binary image to segment
    :param verb: True if messages are expected
    :return: segmented image and number of labels (between 1 and N)
    '''
    def insertConf(cTab, l1, l2):
        for i in range(len(cTab)):
            if l1 in cTab[i] and l2 in cTab[i]:
                return (cTab)  # Label already registered
            elif l1 in cTab[i]:
                cTab[i].append(l2)
                return (cTab)
            elif l2 in cTab[i]:
                cTab[i].append(l1)
                return (cTab)
        cTab.append([l1, l2])
        return (cTab)

    def labEq(lab, confTab):
        for i in range(len(confTab)):
            if lab in confTab[i]:
                return (min(confTab[i]))
        return (lab)

    if verb:
        print('Input image')
        print(img)

    (height, width) = img.shape

    # Labels between 1 and N
    nextLabel = 1
    if img[0][0] == 1:
        img[0][0] = nextLabel  # First pixel
        nextLabel += 1
    for x in range(1, width):  # First row
        if img[0][x] == 1:
            if img[0][x - 1] != 0:
                img[0][x] = img[0][x - 1]
            else:
                img[0][x] = nextLabel
                nextLabel += 1

    confTab = []

    for y in range(1, height):  # Remaining rows
        if img[y][0] == 1:
            if img[y - 1][0] != 0:  # First column
                img[y][0] = img[y - 1][0]
            else:
                img[y][0] = nextLabel
                nextLabel += 1
        else:
            img[y][0] = nextLabel
            nextLabel += 1
        for x in range(1, width):  # Remaining columns
            if img[y][x] == 1:
                if img[y - 1][x] != 0:
                    img[y][x] = img[y - 1][x]
                    if img[y][x - 1] != 0 and img[y][x - 1] != img[y][x]:
                        print('Conflito', img[y][x - 1], img[y][x])
                        confTab = insertConf(confTab, img[y][x - 1], img[y][x])
                elif img[y][x - 1] != 0:
                    img[y][x] = img[y][x - 1]
                else:
                    img[y][x] = nextLabel
                    nextLabel += 1
    if verb:
        print('Labels before equivalencies')
        print(img)

    labDict = {0: 0}
    off = 0
    for l in range(1, nextLabel + 1):
        le = labEq(l, confTab)
        if l != le:
            labDict[l] = labDict[le]
            off += 1
        else:
            labDict[l] = l - off
    if verb:
        print('Conflit table')
        print(confTab)
        print('Number of labels before equivalencies:', nextLabel)
        print('Dictionary')
        print(labDict)
        print('Number of labels after equivalencies:', nextLabel - off - 1)

    for y in range(0, height):
        for x in range(0, width):
            img[y][x] = labDict[img[y][x]]

    if verb:
        print('Labels after equivalencies')
        print(img)

    return (img, nextLabel - off - 1)
