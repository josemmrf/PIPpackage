import os
import random
import io
import sys
import pandas as pd
import statistics
import math
from copy import deepcopy
from shutil import copyfile

filesPath='/Users/josefonseca/PycharmProjects/exame'
examsPath='/Users/josefonseca/PycharmProjects/exame/Exams'
solutionsPath='/Users/josefonseca/PycharmProjects/exame/Solutions'

from checkEmail import checkEmail
from checkEmail import loginEmail

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

#########
# Exame generation functions
#################

def getTxt(vals,images):
    f = open(os.path.join(filesPath,"template 1 teste.txt"), "r")
    s = f.read()
    for i in range(len(vals)):
        s=s.replace(f'<{i+1}>',str(vals[i]))
    for i in range(len(images)):
        s=s.replace(f'<image{i+1}>',images[i])
    return(s)

def randof(list):
    i=random.randint(0,len(list)-1)
    return([list[i]])

def genImage(dim,max):
    s=''
    listV=[]
    for i in range(dim):
        lin=[]
        for j in range(dim):
            v=random.randint(0,max)
            s +="{0:5d}".format(v)
            lin +=[v]
        s +='\n'
        listV +=[lin]
    return(s,listV)

def genVals(studentNumb):
    random.seed(studentNumb)
    vals=[]
    vals += randof([2,3])
    vals += randof([2,3])
    vals += [random.randint(5,20)]
    vals += randof([1, 2, 3])
    vals += randof([1, 2, 3])
    vals += randof([0, 4])
    vals += randof([1, 2, 3])
    vals += [random.randint(520, 620)]
    return(vals)

##########
# Generate exam
####################

def genExam(studentNumb,name):

    try:
        os.mkdir(os.path.join(examsPath,str(studentNumb)))
    except:
        pass
    try:
        os.mkdir(os.path.join(solutionsPath,str(studentNumb)))
    except:
        pass

    os.chdir(os.path.join(examsPath,str(studentNumb)))

    original_stdout=sys.stdout
    with open(str(studentNumb)+'_exame_SS.txt', 'w') as f:
        sys.stdout = f
        img1,imlist1=genImage(5,127)
        img2,imlist2=genImage(6,7)
        while img2==eqHist(imlist2,8,False):
            img2, imlist2 = genImage(6, 7)
        img3, imList3 = genImage(6, 1)
        while not numBits(imList3) in [8, 12]:
            img3, imList3 = genImage(6, 1)
        kerHitMiss = [[-1, 0, -1], [-1, 0, -1], [-1, 0, 1]]
        img4, imList4 = genImage(6, 1)
        while not numBits(hitAndMiss(imList4, kerHitMiss)) in [8, 12]:
            img4, imList4 = genImage(6, 1)

        vals=genVals(studentNumb)
        txt=getTxt(vals,[img1,img2,img3,img4])
        txt=txt.replace('<numero>',str(studentNumb))
        txt=txt.replace('<nome>',name)
        print(txt)
        f.close()
        copyfile(os.path.join(examsPath,str(studentNumb),str(studentNumb)+'_exame_SS.txt'),
                                os.path.join(solutionsPath,str(studentNumb),str(studentNumb)+'_exame_SS.txt'))

    sys.stdout = original_stdout

#### Corretion generation #####

    os.chdir(os.path.join(solutionsPath,str(studentNumb)))
    with open(str(studentNumb)+'_exame_SS_sol.txt', 'w') as f:
        sys.stdout = f
        print('\nAluno:',name,'  Numero:',studentNumb)
    #2 Kernal separado
        print('PERGUNTA 2')
        imRes = sepFilter3x3([2,3,2], [2,6,2], imlist1)
        print('Valor do pixel (2,2):','{:6.3f}'.format(imRes[2][2]))
        print('Valor do pixel (4,3):','{:6.3f}'.format(imRes[3][4]))
    #3 Rotacao
        print('PERGUNTA 3')
        printImg(imlist1)
        xorig, yorig = calcRotCoord(vals[0], vals[1], vals[2])
        res = bilinear(xorig, yorig, imlist1)
        print('Final result', res)
    #4 Filtro Mediana Hibrido 5x5
        print('PERGUNTA 4')
        medianHibrid5x5(vals[3],vals[4],imlist1,True)
        medianHibrid5x5(vals[5],vals[6],imlist1,False)
    #5 Equializacao de histograma
        print('PERGUNTA 5')
        printImg(imlist2)
        eqHist(imlist2,8,True)
    #6 Fecho numa imagem binaria com kernel
        print('PERGUNTA 6')
        ker = [[0, 1, 0], [0, 1, 1], [1, 0, 1]]
        print('Kernel')
        printImg(ker)
        print('Original image')
        printImg(imList3)
        imResD = dilation(imList3, ker)
        print('Dilated image')
        printImg(imResD)
        imResE = erosion(imResD, ker)
        print('Eroded image')
        printImg(imResE)
    # Fecho numa imagem binaria com kernel
#        print('PERGUNTA 6')
#        print('Input binary image')
#        printImg(imList4)
#        print('Hit & Miss kernel')
#        printImg(kerHitMiss)
#        imResHM = hitAndMiss(imList4, kerHitMiss)
#        print('Hit & Miss resulting image')
#        printImg(imResHM)
    # Sensores
        print('PERGUNTA 7')
        readings = [[20.6, 26], [31.1, 192], [32.5, 213], [33.3, 218], [34.2, 274], [39.6, 294], [40.0, 380],
                    [48.3, 413], [48.8, 454], [49.2, 479], [49.0, 500], [58.6, 640], [59.4, 641], [60.5, 668],
                    [63.2, 679], [63.6, 681], [64.1, 732]]

        sensorV = vals[7]
        realV = 55

        upper = 0
        while readings[upper][1] < sensorV:
            upper += 1
        for j in range(upper + 1, len(readings)):
            dif = readings[j][1] - sensorV
            if dif > 0 and dif < readings[upper][1] - sensorV:
                upper = j

        lower = 0
        while readings[lower][1] > sensorV:
            lower += 1
        for j in range(lower + 1, len(readings)):
            dif = sensorV - readings[j][1]
            if dif > 0 and dif < sensorV - readings[lower][1]:
                lower = j

        print('Sensor value:', sensorV)
        print('Lower value:', readings[lower][1], readings[lower][0])
        print('Higher value:', readings[upper][1], readings[upper][0])
        off = (sensorV - readings[lower][1]) / (readings[upper][1] - readings[lower][1])
        print('Offset:', '{:5.3f}'.format(off))
        v = interp(readings[lower][0], readings[upper][0], off)
        print('Absolute error:','{:5.2f}'.format(v - realV))
        print('Relative error:'+'{:5.2f}'.format(abs(v - realV) / realV * 100) + '%')
        f.close()

    sys.stdout = original_stdout

def makeFolders():
    os.chdir(filesPath)
    try:
        os.mkdir('Exams')
    except:
        pass
    try:
        os.mkdir('Solutions')
    except:
        pass

#######
# MAIN PROGRAM
#######################

#insc = pd.read_excel(os.path.join(filesPath,'inscritosSS.xlsx'),engine='openpyxl')
#makeFolders()
#for i in range(len(insc)):
#    genExam(insc['Numero'][i],insc['Nome'][i])

mail = loginEmail('ssfctunl@gmail.com', 'Sensoriais2020')  # 'simfctunl@gmail.com', 'sim576911')
checkEmail(mail)



