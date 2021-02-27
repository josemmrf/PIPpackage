import os
import random
import io
import sys
import pandas as pd
import statistics
import math
from copy import deepcopy
from shutil import copyfile

#########
# Image processing functions
#################
from math import *

########
# Print an image of integer pixels
################
def printImg(img):
    for l in img:
        for p in l:
            print('{0:6d}'.format(p), end='')
        print()

########
# Print an image of float pixels
################
def printImgFloat(img):
    for l in img:
        for p in l:
            print('{:8.2f}'.format(p), end='')
        print()

########
# Pad an image (1,1,1,1) duplicated
################
def imagePadDup1(image):
    res = []
    for i in range(len(image)):
        res.append([image[i][0]] + image[i] + [image[i][-1]])
    res.insert(0, res[0])
    res.insert(-1, res[-1])
    return (res)

########
# Pad an image (2,2,2,2) duplicated
################
def imagePadDup2(image):
    res = []
    for i in range(len(image)):
        res.append([image[i][0]] + [image[i][0]] + image[i] + [image[i][-1]] + [image[i][-1]])
    res.insert(0, res[0])
    res.insert(0, res[0])
    res.insert(-1, res[-1])
    res.insert(-1, res[-1])
    return (res)

########
# Convolute a padded image with a 3x3 kernel
################
def imgC3x3(img, ker):
    tot = 0
    for l in ker:
        tot += sum(l)

    print('Valor da soma do filtro',tot)

    imRes = []
    for y in range(1, len(img) - 1):
        imRes.append([])
        for x in range(1, len(img[y]) - 1):
            s = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    s += img[y + i][x + j] * ker[i + 1][j + 1]
            imRes[y - 1].append(s)

    print('Imagem resultado antes da divisão')
    printImg(imRes)
    imDiv=[[v / tot for v in lin] for lin in imRes]
    print('Imagem resultado depois de dividida')
    printImgFloat(imDiv)
    return ([[v / tot for v in lin] for lin in imRes])

########
# Convolute a unpadded image with a 3x3 kernel
################
def imgConv3x3(img, ker):
    imgPad = imagePadDup1(img)
#    print('Imagem com duplicação de margens')
#    printImg(imgPad)
    imgRes = imgC3x3(imgPad, ker)
    return (imgRes)

########
# Filter the image img with a separated kernel filter col x lin
# Returns: floating point image with the scaled result
################
def sepFilter3x3(col, lin, img):
    ker = []
    for y in range(3):
        ker.append([])
        for x in range(3):
            ker[y].append(col[y] * lin[x])
    print('Kernel completo (a partir do separado)')
    printImg(ker)
    return (imgConv3x3(img, ker))

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
    v1 = statistics.median(l1)

    v2 = statistics.median(l2)
    res = statistics.median([v1, v2, image[y][x]])
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



