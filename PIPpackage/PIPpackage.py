import math
from statistics import median
from math import sin, cos
from copy import deepcopy
import random
import numpy as np
import time
from math import floor, ceil
import matplotlib.pyplot as plt
from PIL import Image
import random

########
# GenImage - generates an numpy array with a random image
################
def genImage(dimX, dimY, max, seed=0):
    '''
    Generates a numpy array with dimX by dimY pixels of type int
    - returns the array filled with values between zero and max
    '''
    if seed != 0:
        random.seed(seed)
    img = np.empty((dimY, dimX), int)
    for x in range(dimX):
        for y in range(dimY):
            img[y, x] = random.randint(0, max)
    return img


########
# imagePad - padding an image an arbitrary number of lines and columns
################
def imagePad(image, padding, mode='edge'):
    ''' imagePad - padding an image
    :param image: NumPy array with the image
    :param padding: list of padding values - yUp, yDown, xLeft, xRight
    :param mode: it can be ‘edge’ (default), ‘symmetric’, 'wrap', 'constant' (padding with zeros)
    :return: NumPy array with padded image
    '''
    yu = padding[2]
    yd = padding[3]
    xl = padding[0]
    xr = padding[1]
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
    dimY = img.shape[0]
    dimX = img.shape[1]
    (dkY, dkX) = kernel.shape
    offX = floor(dkX / 2)
    offY = floor(dkY / 2)

    imRes = np.empty([dimY - 2 * offY, dimX - 2 * offX], int)

    #    print(imRes.shape)
    for y in range(offY, dimY - offY):
        for x in range(offX, dimX - offX):
            s = 0
            for i in range(-offY, offY + 1):
                for j in range(-offX, offX + 1):
                    s += img[y + i][x + j] * kernel[i + offY][j + offX]
            imRes[y - offY][x - offX] = s

    imDiv = imRes / kernel.sum()

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
    offX = floor(dkX / 2)
    offY = floor(dkY / 2)
    imgPad = imagePad(img, [offY, offY, offX, offX])
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
    kernel = np.empty([len(col), len(lin)])
    for y in range(len(col)):
        for x in range(len(lin)):
            kernel[y][x] = col[y] * lin[x]
    #    print('Kernel completo (a partir do separado)')
    #    printImgFloat(kernel)
    return imgConv(img, kernel)


########
# Salt and Pepper Noise
################
def spNoise(img, pb=10):
    '''
    Inserts Salt and Pepper noise in the image
    :param img: Input image
    :param pb: Probability for the "salt" and " pepper" noise
    :return: Noisy image
    '''

    (height, width) = img.shape
    res = np.zeros((height, width), dtype=np.uint8)
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    if pb > 50:
        pb = 50
    elif pb < 10:
        pb = 10

    pb = int(pb / 10)
    black = values[0:pb]
    white = values[10 - pb:10]
    print(white, black)
    for y in range(0, height):
        for x in range(0, width):
            var = random.randint(0, 49)
            if white.__contains__(var):
                res[y][x] = 0
            elif black.__contains__(var):
                res[y][x] = 255
            else:
                res[y][x] = img[y][x]

    return res, 3


########
# Calculate translation destiny coordinates
################
def calcTranslation(x, y, x0, y0):
    '''
    Translation Operation
    :param x: x value position
    :param y: y value position
    :param x0: x offset
    :param y0: y offset
    :return: new coordinates
    '''
    return x + x0, y + y0


########
# Calculate translation source coordinates
################
def calcTranslationCoord(x, y, x0, y0):
    '''
    Translation Operation
    :param x: x value position
    :param y: y value position
    :param x0: x offset
    :param y0: y offset
    :return: new coordinates
    '''
    return x0 - x, y0 - y


########
# Calculate rotation destiny coordinates (angle in degrees)
################
def calcRot(x, y, ang):
    '''
    Rotation Operation
    :param x: x value position
    :param y: y value position
    :param ang: rotation angle
    :return: new coordinates
    '''
    ang = ang / 180 * np.pi
    x0 = x * cos(ang) + y * sin(ang)
    y0 = -x * sin(ang) + y * cos(ang)
    return x0, y0


########
# Calculate rotation source coordinates (angle in degrees)
################
def calcRotCoord(x, y, ang):
    '''
    Rotation Operation
    :param x: x value position
    :param y: y value position
    :param ang: rotation angle
    :return: new coordinates
    '''
    ang = ang / 180 * np.pi
    xorig = x * cos(ang) - y * sin(ang)
    yorig = x * sin(ang) + y * cos(ang)
    return (xorig, yorig)


########
# Calculate zoom source coordinates
################
def calcZoom(x, y, s):
    '''
    Zoom Operation
    :param x: x value position
    :param y: y value position
    :param s: scaling factor
    :return: new coordinates
    '''
    return x / s, y / s


########
# Calculate zoom destiny coordinates
################
def calcZoomCoord(x, y, s):
    '''
    Zoom Operation
    :param x: x value position
    :param y: y value position
    :param s: scaling factor
    :return: new coordinates
    '''
    return x * s, y * s


########
# Linear interpolation
################
def interp(a, b, off, verb=False):
    '''
    Linear interpolation
    :param a: First value
    :param b: Second value
    :param off: Offset between first and second value
    :param verb: True if messages are expected
    :return: exact interpolation value (not rounded)
    '''
    res = (1 - off) * a + b * off
    if verb:
        print('Interpolating between', a, 'and', b, 'at', off, '=', res)

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
        print('Values to consider:', v1, v2, v3, v4)
    vv1 = interp(v1, v2, xorig - x1, verb)
    vv2 = interp(v3, v4, xorig - x1, verb)
    res = interp(vv1, vv2, yorig - y1, verb)

    if res < 0:
        res = 0
    elif res > 255:
        res = 255

    return round(res)


########
# Bi-cubic interpolation Aux
################
def bicubic_aux(values, off, verb=False):
    '''
    Bi-cubic interpolation Auxiliary Function
    :param values: values for interpolation
    :param off: Offset between first and second value
    :param verb: True if messages are expected
    :return: result of interpolated values
    '''

    res = values[1] + off * (0.5 * values[2] - 0.5 * values[0]) + np.power(off, 2) * (
            -0.5 * values[3] + 2 * values[2] - 2.5 * values[1]
            + values[0]) + np.power(off, 3) * (
                  0.5 * values[3] - 1.5 * values[2] + 1.5 * values[1] - 0.5 * values[0])
    if verb:
        print('Interpolating between', values[0], values[1], values[2], 'and', values[3], 'at', off, '=', res)
    return res


########
# Bicubic interpolation
################
def bicubic(xorig, yorig, img, verb=False):
    '''
    Bi-cubic interpolation
    :param xorig: X coordinate
    :param yorig: Y coordinate
    :param img: Image to interpolate
    :param verb: True if messages are expected
    :return: rounded value of resulting bi-cubic interpolation
    '''
    if verb:
        print('Bi-cubic to find', xorig, yorig)

    x1 = floor(xorig)
    y1 = floor(yorig)

    listoff = [[-1, -1], [-1, 0], [-1, 1], [-1, 2], [0, -1], [0, 0], [0, 1], [0, 2], [1, -1], [1, 0], [1, 1], [1, 2],
               [2, -1], [2, 0], [2, 1], [2, 2]]

    values = [img[y1 + off[0]][x1 + off[1]] for off in listoff]

    if verb:
        print('Values to consider:', values)

    vvs = [bicubic_aux(values[0:4], xorig - x1, verb),
           bicubic_aux(values[4:8], xorig - x1, verb),
           bicubic_aux(values[8:12], xorig - x1, verb),
           bicubic_aux(values[12:16], xorig - x1, verb)]
    res = bicubic_aux(vvs[0:4], yorig - y1, verb)

    if res < 0:
        res = 0
    elif res > 255:
        res = 255

    return round(res)


########
# Translate Image
################
def TranslateImg(img, x0, y0, verb=False):
    '''
    Translation Operation
    :param img: Image to apply the operation translation
    :param x0: x offset
    :param y0: y offset
    :param verb: True if messages are expected
    :return: Translated image
    '''

    height = img.shape[0]
    width = img.shape[1]
    res = np.zeros((height, width), dtype=np.uint8)

    old_time = time.time()

    for y in range(0, height):
        for x in range(0, width):
            (a, b) = calcTranslation(x, y, x0, y0)

            if verb:
                print("New coordinates: ", a, b)

            if 0 < a < width - 1 and 0 < b < height - 1:
                res[b][a] = img[b][a]

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Rotate Image
################
def RotateImg(img, angle, sel='BL', verb=False):
    '''
    Rotation Operation
    :param img: Image to apply the operation rotation
    :param angle: Rotation angle in  degrees
    :param sel: Selects bilinear (BL) or bicubic (BC) interpolation
    :param verb: True if messages are expected
    :return: Rotated image
    '''

    height = img.shape[0]
    width = img.shape[1]
    res = np.zeros((height, width), dtype=np.uint8)

    old_time = time.time()

    for y in range(0, height):
        for x in range(0, width):
            (a, b) = calcRot(x, y, angle)
            if 0 < a < width - 2 and 0 < b < height - 2:
                if ceil(a) - a != 0 or ceil(b) - b != 0:
                    if sel == 'BL':
                        res[y][x] = bilinear(a, b, img, verb)
                    if sel == 'BC':
                        res[y][x] = bicubic(a, b, img, verb)
                    if sel == 'NN':
                        res[y][x] = img[round(b)][round(a)]
                else:
                    res[y][x] = img[b][a]

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Resize Image
################
def ResizeImg(img, s, sel='BL', verb=False):
    '''
    Resizing Operation
    :param img: Image to apply the operation rotation
    :param s: Resizing factor
    :param sel: Selects bilinear (BL) or bicubic (BC) interpolation
    :param verb: True if messages are expected
    :return: Resized image
    '''

    height = img.shape[0]
    width = img.shape[1]
    res = np.zeros((height, width), dtype=np.uint8)

    old_time = time.time()

    for y in range(0, height):
        for x in range(0, width):
            (a, b) = calcZoom(x, y, s)
            if s <= 1:
                if 0 < a < width - 2 and 0 < b < height - 2:
                    if sel == 'BL':
                        res[y][x] = bilinear(a, b, img, verb)
                    if sel == 'BC':
                        res[y][x] = bicubic(a, b, img, verb)
                    if sel == 'NN':
                        res[y][x] = img[round(b)][round(a)]
            else:
                if sel == 'BL':
                    res[y][x] = bilinear(a, b, img, verb)
                if sel == 'BC':
                    res[y][x] = bicubic(a, b, img, verb)
                if sel == 'NN':
                    res[y][x] = img[round(b)][round(a)]

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Mean Filter Type A
################
def meanFilterA(image, dimX, dimY, verb=False):
    '''
    Mean filter
    :param image: image to be filtered
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :param verb: True if messages are expected
    :return: returns the result image
    '''

    count_sum = 0
    count_div = 0
    height = image.shape[0]
    width = image.shape[1]
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((height - (offY * 2), width - (offX * 2)), dtype=np.ubyte)

    old_time = time.time()

    for y in range(offY, height - offY):
        for x in range(offX, width - offX):
            values = [image[y + i][x + j] for i in range(-offY, offY + 1) for j in range(-offX, offX + 1)]
            mean = 0
            if verb:
                print('Values to consider:', values)
            for value in values:
                mean += value
            count_sum += dimX * dimY
            res[y - offY][x - offX] = mean / (dimX * dimY)
            count_div += 1
            if verb:
                print('Result: ', res[y - offY][x - offX])

    new_time = time.time()

    if verb:
        print('Number of sums: ', count_sum)
        print('Number of div: ', count_div)

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Mean Filter Type B
################
def meanFilterB(image, dimX, dimY, verb=False):
    '''
    Mean filter
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :param image: image to be filtered
    :param verb: True if messages are expected
    :return: returns the result image
    '''

    count_div = 0
    height = image.shape[0]
    width = image.shape[1]
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((height - (offY * 2), width - (offX * 2)), dtype=np.uint)
    res2 = np.zeros((height - (offY * 2), width - (offX * 2)), dtype=np.ubyte)

    old_time = time.time()

    values = [image[offY + i][offX + j] for i in range(-offY, offY + 1) for j in range(-offX, offX + 1)]
    mean = 0
    if verb:
        print('Values to consider:', values)
    for value in values:
        mean += value
    res[0][0] = mean
    count_sum = (dimX * dimY)
    if verb:
        print('Result: ', int(res[0][0] / (dimX * dimY)))

    for x in range(offX + 1, width - offX):
        values_sub = [image[offY + i][x - (offX + 1)] for i in range(-offY, offY + 1)]
        values_add = [image[offY + j][x + offX] for j in range(-offY, offY + 1)]
        sub = 0
        add = 0
        if verb:
            print('Values to subtract:', values_sub)
            print('Values to add:', values_add)
        for value in values_sub:
            sub += value
        for value in values_add:
            add += value
        mean = mean + add - sub
        count_sum += (dimX + dimY)
        res[0][x - offX] = mean
        if verb:
            print('Result: ', int(res[0][x - offX] / (dimX * dimY)))

    for y in range(offY + 1, height - offY):
        values_sub = [image[y - (offY + 1)][offX + i] for i in range(-offX, offX + 1)]
        values_add = [image[y + offY][offX + j] for j in range(-offX, offX + 1)]
        sub = 0
        add = 0
        if verb:
            print('Values to subtract:', values_sub)
            print('Values to add:', values_add)
        for value in values_sub:
            sub += value
        for value in values_add:
            add += value
        mean = res[y - offY - 1][0]
        mean = mean + add - sub
        count_sum += (dimX + dimY)
        res[y - offY][0] = mean
        if verb:
            print('Result: ', int(res[y - offY][0] / (dimX * dimY)))

        for x in range(offX + 1, width - offX):
            values_sub = [image[y + i][x - (offX + 1)] for i in range(-offY, offY + 1)]
            values_add = [image[y + j][x + offX] for j in range(-offY, offY + 1)]
            sub = 0
            add = 0
            if verb:
                print('Values to subtract:', values_sub)
                print('Values to add:', values_add)
            for value in values_sub:
                sub += value
            for value in values_add:
                add += value
            mean = mean + add - sub
            count_sum += (dimX + dimY)
            res[y - offY][x - offX] = mean
            if verb:
                print('Result: ', int(res[y - offY][x - offX] / (dimX * dimY)))

    for y in range(0, (height - (offY * 2))):
        for x in range(0, (width - (offX * 2))):
            # res[y][x] = res[y][x] / (dimX * dimY)
            res2[y][x] = res[y][x] / (dimX * dimY)
            count_div += 1

    new_time = time.time()

    if verb:
        print('Number of sums: ', count_sum)
        print('Number of div: ', count_div)

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res2


########
# Mean Filter Type C
################
def meanFilterC(image, dimX, dimY, verb=False):
    '''
    Mean filter
    :param image: image to be filtered
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :param verb: True if messages are expected
    :return: returns the result image
    '''

    count_div = 0
    height = image.shape[0]
    width = image.shape[1]
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    conneroff = [[-(offY + 1), -(offX + 1)], [-(offY + 1), offX], [offY, -(offX + 1)], [offY, offX]]
    res = np.zeros((height - (offY * 2), width - (offX * 2)), dtype=np.uint)
    res2 = np.zeros((height - (offY * 2), width - (offX * 2)), dtype=np.ubyte)

    old_time = time.time()

    values = [image[offY + i][offX + j] for i in range(-offY, offY + 1) for j in range(-offX, offX + 1)]
    mean = 0
    if verb:
        print('Values to consider:', values)
    for value in values:
        mean += value
    res[0][0] = mean
    count_sum = (dimX * dimY)
    if verb:
        print('Result: ', int(res[0][0] / (dimX * dimY)))

    for x in range(offX + 1, width - offX):
        values_sub = [image[offY + i][x - (offX + 1)] for i in range(-offY, offY + 1)]
        values_add = [image[offY + j][x + offX] for j in range(-offY, offY + 1)]
        sub = 0
        add = 0
        if verb:
            print('Values to subtract:', values_sub)
            print('Values to add:', values_add)
        for value in values_sub:
            sub += value
        for value in values_add:
            add += value
        mean = mean + add - sub
        count_sum += (dimX + dimY)
        res[0][x - offX] = mean
        if verb:
            print('Result: ', int(res[0][x - offX] / (dimX * dimY)))

    for y in range(offY + 1, height - offY):
        values_sub = [image[y - (offY + 1)][offX + i] for i in range(-offX, offX + 1)]
        values_add = [image[y + offY][offX + j] for j in range(-offX, offX + 1)]
        sub = 0
        add = 0
        if verb:
            print('Values to subtract:', values_sub)
            print('Values to add:', values_add)
        for value in values_sub:
            sub += value
        for value in values_add:
            add += value
        mean = res[y - offY - 1][0]
        mean = mean + add - sub
        count_sum += (dimX + dimY)
        res[y - offY][0] = mean
        if verb:
            print('Result: ', int(res[y - offY][0] / (dimX * dimY)))

        for x in range(offX + 1, width - offX):
            m1 = res[y - (offY + 1)][x - offX]
            m2 = res[y - (offY + 1)][x - (offX + 1)]
            m3 = res[y - offY][x - (offX + 1)]
            conner_values = [image[y + off[0]][x + off[1]] for off in conneroff]
            if verb:
                print('Values to subtract:', m2, conner_values[0:2])
                print('Values to add:', m1, m3, conner_values[3])
            mean = m1 - m2 + m3 + conner_values[0] - conner_values[1] - conner_values[2] + conner_values[3]
            count_sum += 6
            res[y - offY][x - offX] = mean
            if verb:
                print('Result: ', int(res[y - offY][x - offX] / (dimX * dimY)))

    for y in range(0, height - (offX * 2)):
        for x in range(0, width - (offX * 2)):
            res2[y][x] = res[y][x] / (dimX * dimY)
            count_div += 1

    new_time = time.time()

    if verb:
        print('Number of sums: ', count_sum)
        print('Number of div: ', count_div)
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res2


########
# Generic Nonuniform Mean Filter
################
def meanNonUniFilter(coefficients, image, dimX, dimY, verb=False):
    '''
    Mean filter with coefficients associated
    :param coefficients: coefficient list
    :param image: image to be filtered
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :param verb: True if messages are expected
    :return: returns the result image
    '''

    height = image.shape[0]
    width = image.shape[1]
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((height - (offY * 2), width - (offX * 2)), dtype=np.uint8)
    c = 0

    old_time = time.time()

    for coff in coefficients:
        c = c + coff

    for y in range(offY, height - offY):
        for x in range(offX, width - offX):
            values = [image[y + i][x + j] for i in range(-offY, offY + 1) for j in range(-offX, offX + 1)]
            mean = 0
            if verb:
                print('Values do consider: ', values)
                print('Coefficients: ', coefficients)
            newvalues = [value * coefficient for value, coefficient in zip(values, coefficients)]

            for value in newvalues:
                mean += value

            res[y - offY][x - offX] = mean / c
            if verb:
                print('New values after coefficients to consider: ', newvalues)
                print('Result: ', res[y - offY][x - offX])

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Nonlinear Mean Filter
################
def meanNonLinearFilter(image, dimX, dimY, t, verb=False):
    '''
    Nonlinear mean filter
    :param image: image to be filtered
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :param t: mean value threshold
    :param verb: True if messages are expected
    :return: returns the result image
    '''

    height = image.shape[0]
    width = image.shape[1]
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((height - (offY * 2), width - (offX * 2)), dtype=np.uint8)

    old_time = time.time()

    for y in range(offY, height - offY):
        for x in range(offX, width - offX):
            values = [image[y + i][x + j] for i in range(-offY, offY + 1) for j in range(-offX, offX + 1)]
            mean = 0
            if verb:
                print('Values to consider:', values)
            for value in values:
                mean += value
            if abs(image[y - offY][x - offX] - mean / (dimX * dimY)) < t:
                res[y - offY][x - offX] = mean / (dimX * dimY)
            else:
                res[y - offY][x - offX] = image[y - offY][x - offX]
            if verb:
                print('Value to compare to threshold: ', int(abs(image[y - offY][x - offX] - mean / (dimX * dimY))))
                print('Result: ', res[y - offY][x - offX])

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Slow BubbleSort
################
def bubblesortA(unsorted, mode):
    '''
    Ascending or Descending sorter
    :param mode: ascending (A) or descending (D) mode
    :param unsorted: list to be sorted
    :return: returns sorted list
    '''

    i = 0
    res = np.zeros(len(unsorted), dtype=np.uint8)

    if mode == 'A':
        while len(unsorted) > 0:
            value = unsorted[0]
            for x in range(0, len(unsorted)):
                if value >= unsorted[x]:
                    value = unsorted[x]
                    index = x
                else:
                    continue
            res[i] = unsorted.pop(index)
            i += 1

    if mode == 'D':
        while len(unsorted) > 0:
            value = unsorted[0]
            for x in range(0, len(unsorted)):
                if unsorted[x] >= value > 0:
                    value = unsorted[x]
                    index = x
                else:
                    continue
            res[i] = unsorted.pop(index)
            i += 1

    return res


########
# Fast BubbleSort
################
def bubblesortB(unsorted, dimX, dimY, mode):
    '''
    Ascending or Descending sorter
    :param unsorted: list to be sorted
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :param mode: ascending (A) or descending (D) mode
    :return: returns sorted list
    '''

    i = 0
    res = np.zeros(int((dimX * dimY) / 2 + 1), dtype=np.uint8)

    if mode == 'A':
        while len(unsorted) > int((dimX * dimY) / 2):
            value = unsorted[0]
            for x in range(0, len(unsorted)):
                if value >= unsorted[x]:
                    value = unsorted[x]
                    index = x
                else:
                    continue
            res[i] = unsorted.pop(index)
            i += 1

    if mode == 'D':
        while len(unsorted) > int((dimX * dimY) / 2):
            value = unsorted[0]
            for x in range(0, len(unsorted)):
                if unsorted[x] >= value > 0:
                    value = unsorted[x]
                    index = x
                else:
                    continue
            res[i] = unsorted.pop(index)
            i += 1

    return res


########
# MinMax Algorithm
################
def minmax(unsorted):
    '''
    MinMax Sorter
    :param unsorted: list to be sorted
    :return: returns sorted list
    '''

    k = len(unsorted) - 1

    for i in range(0, int(len(unsorted) / 2)):
        for j in range(i, len(unsorted) - i):
            if unsorted[j] > unsorted[k]:
                tmp = unsorted[j]
                unsorted[j] = unsorted[k]
                unsorted[k] = tmp
            if unsorted[i] > unsorted[j]:
                tmp = unsorted[i]
                unsorted[i] = unsorted[j]
                unsorted[j] = tmp
        k -= 1

    return unsorted


########
# Median Filter
################
def medianFilter(image, dimX, dimY, sel='SB', verb=False):
    '''
    Median filter
    :param image: image to be filtered
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :param sel: Selects the sorting algorithm (SB = Slow BubleSort; FB = Fast BubleSort; MM = MinMax)
    :param verb: True if messages are expected
    :return: returns the result image
    '''

    height = image.shape[0]
    width = image.shape[1]
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((height - (offY * 2), width - (offX * 2)), dtype=np.uint8)

    old_time = time.time()

    for y in range(offY, height - offY):
        for x in range(offX, width - offX):
            values = [image[y + i][x + j] for i in range(-offY, offY + 1) for j in range(-offX, offX + 1)]

            if verb:
                print('Values to consider:', values)
            if sel == 'SB':
                sorted_values = bubblesortA(values, 'A')
            if sel == 'FB':
                sorted_values = bubblesortB(values, dimX, dimY, 'A')
            if sel == 'MM':
                sorted_values = minmax(values)
            if verb:
                print('Sorted values:', sorted_values)
                print('Median value: ', sorted_values[int((dimX * dimY) / 2)])

            res[y - offY][x - offX] = sorted_values[int((dimX * dimY) / 2)]

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res, format(new_time - old_time, '.3f')


########
# Mode Filter
################
def modeFilter(image, dimX, dimY, verb=False):
    '''
    Mode filter
    :param image: image to be filtered
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :param verb: True if messages are expected
    :return: returns the result image
    '''

    def exists(p, tab):
        for j in range(0, len(tab)):
            if p in tab[j]:
                return 0
        return -1

    height = image.shape[0]
    width = image.shape[1]
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((height - (offY * 2), width - (offX * 2)), dtype=np.uint8)

    old_time = time.time()

    for y in range(offY, height - offY):
        for x in range(offX, width - offX):
            values = [image[y + i][x + j] for i in range(-offY, offY + 1) for j in range(-offX, offX + 1)]
            table = []
            if verb:
                print('Values to consider: ', values)
            for level in values:
                count = 0
                if exists(level, table) == -1:
                    for pixel in values:
                        if pixel == level:
                            count += 1
                    table.append([level, count])
            if verb:
                print('Number of pixel occurrences in kernel: ', table)
            for entry in table:
                if count <= entry[1]:
                    count = entry[1]
                    value = entry[0]
            if verb:
                print('Pixel with most occurrences: ', value)

            res[y - offY][x - offX] = value

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res



########
# Differentiation Filter
################
def difFilter(image, verb=False):
    '''
    Differentiation filter
    :param image: image to be filtered
    :param verb: True if messages are expected
    :return: returns the result image
    '''

    listoff = [[0, 0], [0, 1], [1, 0]]
    height = image.shape[0]
    width = image.shape[1]
    res = np.zeros((height, width), dtype=np.uint8)

    old_time = time.time()

    for y in range(0, height - 1):
        for x in range(0, width - 1):
            values = [image[y + off[0]][x + off[1]] for off in listoff]
            if verb:
                print('Values to consider: ', values)
            res[y][x] = abs(values[0] - values[1]) + abs(values[0] - values[2])
            if verb:
                print('Result: ', res[y][x])

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Roberts Filter
################
def robertsFilter(image, verb=False):
    '''
    Roberts filter
    :param image: image to be filtered
    :param verb: True if messages are expected
    :return: returns the result image
    '''

    listoff = [[0, 0], [0, 1], [1, 0], [1, 1]]
    height = image.shape[0]
    width = image.shape[1]
    res = np.zeros((height, width), dtype=np.uint8)

    old_time = time.time()

    for y in range(0, height - 1):
        for x in range(0, width - 1):
            values = [image[y + off[0]][x + off[1]] for off in listoff]
            if verb:
                print('Values to consider: ', values)
            res[y][x] = abs(values[0] - values[3]) + abs(values[1] - values[2])
            if verb:
                print('Result: ', res[y][x])

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Prewitt Filter
################
def prewittFilter(image, verb=False):
    '''
    Prewitt filter
    :param image: image to be filtered
    :param verb: True if messages are expected
    :return: returns the result image
    '''

    listoffx = [[-1, -1], [1, -1], [-1, 1], [1, 1], [0, -1], [0, 1]]
    listoffy = [[-1, -1], [-1, 1], [-1, 0], [1, -1], [1, 1], [1, 0]]
    height = image.shape[0]
    width = image.shape[1]
    res = np.zeros((height, width), dtype=np.uint8)

    old_time = time.time()

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            valuesx = [image[y + off[0]][x + off[1]] for off in listoffx]
            valuesy = [image[y + off[0]][x + off[1]] for off in listoffy]
            if verb:
                print('X values to consider: ', valuesx)
                print('Y values to consider: ', valuesy)
            sx = abs((valuesx[0] + valuesx[1] + valuesx[4]) - (valuesx[2] + valuesx[3] + valuesx[5]))
            sy = abs((valuesy[3] + valuesy[4] + valuesy[5]) - (valuesy[0] + valuesy[1] + valuesy[2]))
            res[y - 1][x - 1] = sx + sy
            if verb:
                print('X Operator: ', sx)
                print('Y Operator: ', sy)
                print('Result: ', res[y - 1][x - 1])

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Sobel Filter
################
def sobelFilter(image, verb=False):
    '''
    Sobel filter
    :param image: image to be filtered
    :param verb: True if messages are expected
    :return: returns the result image
    '''

    listoffx = [[-1, -1], [1, -1], [-1, 1], [1, 1], [0, -1], [0, 1]]
    listoffy = [[-1, -1], [-1, 1], [-1, 0], [1, -1], [1, 1], [1, 0]]
    height = image.shape[0]
    width = image.shape[1]
    res = np.zeros((height, width), dtype=np.uint8)

    old_time = time.time()

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            valuesx = [image[y + off[0]][x + off[1]] for off in listoffx]
            valuesy = [image[y + off[0]][x + off[1]] for off in listoffy]
            if verb:
                print('X values to consider: ', valuesx)
                print('Y values to consider: ', valuesy)
            sx = abs((valuesx[0] + valuesx[1] + valuesx[4] * 2) - (valuesx[2] + valuesx[3] + valuesx[5] * 2))
            sy = abs((valuesy[3] + valuesy[4] + valuesy[5] * 2) - (valuesy[0] + valuesy[1] + valuesy[2] * 2))
            res[y - 1][x - 1] = sx + sy
            if verb:
                print('X Operator: ', sx)
                print('Y Operator: ', sy)
                print('Result: ', res[y - 1][x - 1])

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Colour Median Filter
################
def colourMedianFilter(image, dimX, dimY, verb=False):
    '''
    Colour Median Filter
    :param image: image to be filtered
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :param verb: True if messages are expected
    :return: returns the result image
    '''

    return


########
# K-Nearest Neighbour Filter
################
def kNearestFilter(image, dimX, dimY, k, verb=False):
    '''
    K-Nearest Neighbour Filter
    :param image: image to be filtered
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :param k: number of neighbours
    :param verb: True if messages are expected
    :return: returns the result image
    '''

    height = image.shape[0]
    width = image.shape[1]
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((height - (offY * 2), width - (offX * 2)), dtype=np.uint8)

    old_time = time.time()

    for y in range(offY, height - offY):
        for x in range(offX, width - offX):
            values = [image[y + i][x + j] for i in range(-offY, offY + 1) for j in range(-offX, offX + 1)]
            table = []
            diffs = []
            pixels = []
            mean = 0
            if verb:
                print('Center pixel: ', image[y][x])
                print('Values to consider: ', values)

            for i in range(0, len(values)):
                table.append([values[i], abs(image[y][x] - values[i])])

            for i in range(0, k):
                count = 0
                diff = 256
                for entry in table:
                    if entry[1] < diff:
                        diff = entry[1]
                        pixel = entry[0]
                        index = count
                    count += 1
                diffs.append(diff)
                pixels.append(pixel)
                table.pop(index)

            if verb:
                print('Closest values to center pixel: ', pixels)

            for value in pixels:
                mean += value

            res[y - offY][x - offX] = mean / k

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res, format(new_time - old_time, '.3f')


########
# Sigma Filter
################
def sigmaFilter(image, dimX, dimY, s, verb=False):
    '''
    Sigma Filter
    :param image: image to be filtered
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :param s: standard variation
    :param verb: True if messages are expected
    :return: returns the result image
    '''

    height = image.shape[0]
    width = image.shape[1]
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((height - (offY * 2), width - (offX * 2)), dtype=np.uint8)

    old_time = time.time()

    for y in range(offY, height - offY):
        for x in range(offX, width - offX):
            values = [image[y + i][x + j] for i in range(-offY, offY + 1) for j in range(-offX, offX + 1)]
            if verb:
                print('Values to consider: ', values)
            tab = []
            mean = 0

            for i in range(0, len(values)):
                if image[y][x] - 2 * s <= values[i] <= image[y][x] + 2 * s:
                    tab.append(values[i])
            if verb:
                print('Values that respect the interval: ', tab)

            for entry in tab:
                mean += entry

            res[y - offY][x - offX] = mean / len(tab)

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Integral Image
################
def intImg(image):
    '''
    Integral Image
    :param image: image to process
    :return: returns the integral image
    '''
    height = image.shape[0]
    width = image.shape[1]
    res = image
    var = 0
    for x in range(0, width):
        for y in range(0, height):
            var = var + image[x][y]
            res[x][y] = var


########
# Hybrid 5x5 median filter value for a pixel
################
def medianHybrid5x5(img, sel='SB', verb=False):
    '''
    Hybrid median filter
    :param img: Image to be filtered
    :param sel: Selects the sorting algorithm (SB = Slow BubleSort; FB = Fast BubleSort; MM = MinMax)
    :param verb: True if messages are expected
    :return:
    '''

    diag = [[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2], [2, -2], [1, -1], [-1, 1], [-2, 2]]
    cross = [[0, -2], [0, -1], [0, 0], [0, 1], [0, 2], [-2, 0], [-1, 0], [1, 0], [2, 0]]
    height = img.shape[0]
    width = img.shape[1]
    res = np.zeros((height, width), dtype=np.uint8)

    def getValues(a, b, off, image):
        v = np.zeros([len(off)], dtype=np.uint8)
        for i in range(len(off)):
            v[i] = image[b + off[i][1]][a + off[i][0]]
        return v

    old_time = time.time()

    for y in range(2, height-2):
        for x in range(2, width-2):
            l1 = list(getValues(x, y, diag, img))
            l2 = list(getValues(x, y, cross, img))

            if sel == 'SB':
                v1 = bubblesortA(l1, 'A')
                v2 = bubblesortA(l2, 'A')
                v3 = bubblesortA(list((v1[4], v2[4], img[y][x])), 'A')
                res[y][x] = v3[1]
            if sel == 'FB':
                v1 = bubblesortB(l1, 3, 3, 'A')
                v2 = bubblesortB(l2, 3, 3, 'A')
                v3 = bubblesortB(list((v1[4], v2[4], img[y][x])), 3, 1, 'A')
                res[y][x] = v3[1]
            if sel == 'MM':
                v1 = minmax(l1)
                v2 = minmax(l2)
                v3 = minmax(list((v1[4], v2[4], img[y][x])))
                res[y][x] = v3[1]

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Histogram equalization
################
def eqHist(img, maxLevel, verb=False):
    '''
    Histogram equalization
    :param img: image to equalize
    :param maxLevel: maximum level of gray
    :param verb: True if messages are expected
    :return: equalized image
    '''
    hist = {v: 0 for v in range(maxLevel + 1)}  # Initialize histogram with zero for all the levels
    tot = 0

    old_time = time.time()

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            hist[round(img[y][x], 0)] = hist[round(img[y][x], 0)] + 1
            tot += 1

    if verb:
        print('Histogram', hist)

    accProb = hist.copy()
    prev = 0
    for i in range(len(accProb)):
        prob = accProb[i] / tot
        accProb[i] = prob + prev
        prev += prob

    if verb:
        print('Accumulated probabilities: ', end='')
        for i in range(len(accProb)):
            print("{:.3f}".format(accProb[i]), end='  ')
        print()

    step = 1 / (maxLevel + 1)
    newLevels = accProb.copy()

    for i in range(len(newLevels)):
        newLevels[i] = ceil(newLevels[i] / step) - 1

    if verb:
        print('Transformation table', newLevels)

    imgRes = deepcopy(img)

    for j in range(imgRes.shape[0]):
        for i in range(imgRes.shape[1]):
            imgRes[j][i] = newLevels[imgRes[j][i]]

    if verb:
        print('Resulting image')
        printImg(imgRes)

    new_time = time.time()

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
# Otsu Algorithm
################
def otsuMethod(img, maxLevel=255, verb=False):
    '''
    Otsu Algorithm
    :param img: image to analyse
    :param maxLevel: maximum pixel value in image
    :param verb: True if messages are expected
    :return: Binarized image
    '''

    hist = {v: 0 for v in range(maxLevel + 1)}  # Initialize histogram dictionary with zero for all the levels
    tot = 0
    height = img.shape[0]
    width = img.shape[1]
    res = np.zeros((height, width), dtype=np.uint8)
    old_time = time.time()

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            hist[round(img[y][x], 0)] = hist[round(img[y][x], 0)] + 1
            tot += 1
    if verb:
        print('Histogram', hist)

    probs = hist.copy()

    for i in range(len(probs)):
        prob = probs[i] / tot
        probs[i] = prob

    if verb:
        print('Probabilities: ', end='')
        for i in range(len(probs)):
            print("{:.3f}".format(probs[i]), end='  ')
        print()

    q1 = probs[0]
    q2 = 1 - q1
    u1 = list(probs.keys())[0]
    u2 = 0
    for i in range(1, len(probs)):
        u2 = (u2 + i * probs[i])

    t = 0

    if u1 > 0:
        maxV = q1 * q2 * pow(u1 / q1 - u2 / q2, 2)
    else:
        maxV = 0

    if verb:
        print('Current threshold= ', t)
        print('Q1= ', q1)
        print('Q2= ', q2)
        print('U1= ', u1 / q1)
        print('U2= ', u2 / q2)
        print('Inter-class variation= ', maxV)

    for i in range(1, len(probs)):
        q1 = q1 + probs[i]
        if q1 == 0:
            continue

        q2 = q2 - probs[i]
        if q2 == 0:
            break

        u1 = u1 + i * probs[i]
        u2 = u2 - i * probs[i]

        var = q1 * q2 * pow(u1 / q1 - u2 / q2, 2)

        if verb:
            print('Current threshold= ', i)
            print('Q1= ', q1)
            print('Q2= ', q2)
            print('U1= ', u1 / q1)
            print('U2= ', u2 / q2)
            print('Inter-class variation= ', var)

        if var > maxV:
            t = i
            maxV = var

    if verb:
        print('Optimal threshold value= ', t)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y][x] < t:
                res[y][x] = 0
            else:
                res[y][x] = 255

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# C-Means Algorithm
################
def cmeansMethod(img, maxLevel=255, verb=False):
    '''
    C-Means Algorithm
    :param img: image to analyse
    :param maxLevel: maximum pixel value in image
    :param verb: True if messages are expected
    :return: Binarized image
    '''

    hist = {v: 0 for v in range(maxLevel + 1)}  # Initialize histogram dictionary with zero for all the levels
    tot = 0
    height = img.shape[0]
    width = img.shape[1]
    res = np.zeros((height, width), dtype=np.uint8)

    old_time = time.time()
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            hist[round(img[y][x], 0)] = hist[round(img[y][x], 0)] + 1
            tot += 1
    if verb:
        print('Histogram', hist)

    old_t = round(len(hist) / 2)

    for k in range(len(hist)):
        value1 = 0
        value2 = 0
        value3 = 0
        value4 = 0

        for i in range(old_t):
            value1 = value1 + i * hist[i]
        for i in range(old_t):
            value2 = value2 + hist[i]
        for j in range(old_t + 1, maxLevel):
            value3 = value3 + j * hist[j]
        for j in range(old_t + 1, maxLevel):
            value4 = value4 + hist[j]

        if verb:
            print('Trying current threshold: ', old_t)
            print('Values for the operation: ', value1, value2, value3, value4)

        new_t = round(1 / 2 * (value1 / value2 + value3 / value4))

        if new_t == old_t:
            break
        else:
            old_t = new_t

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y][x] < old_t:
                res[y][x] = 0
            else:
                res[y][x] = 255

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Wellner Algorithm
################
def wellnerMethodA(image, r=1 / 8, t=15, verb=False):
    '''
    Wellner Algorithm
    :param image: image to binarize
    :param r: ratio of pixel to process according to image width
    :param t: percentage do classify the pixel as black or white
    :param verb: True if messages are expected
    :return: returns the binary image
    '''

    height = image.shape[0]
    width = image.shape[1]
    res = np.zeros((height, width), dtype=np.uint8)
    values = []
    s = round(width * r)

    if verb:
        print('Number of pixels to consider: ', s)

    old_time = time.time()

    for y in range(0, height):
        for x in range(0, width):
            values.append(image[y, x])
            mean = 0
            for i in values:
                mean = mean + i

            mean = mean / len(values)

            if image[y, x] < mean * (1 - t / 100):
                res[y, x] = 0
            else:
                res[y, x] = 255

            if verb:
                print('Current pixel value: ', image[y, x])
                print('Array of the latest values: ', values)
                print('Mean of the latest values: ', mean)

            if len(values) == s:
                values.pop(0)
            else:
                continue

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Wellner Algorithm
################
def wellnerMethodB(image, r=1 / 8, t=15, verb=False):
    '''
    Wellner Algorithm
    :param image: image to binarize
    :param r: ratio of pixel to process according to image width
    :param t: percentage do classify the pixel as black or white
    :param verb: True if messages are expected
    :return: returns the binary image
    '''

    height = image.shape[0]
    width = image.shape[1]
    res = np.zeros((height, width), dtype=np.uint8)
    values = []
    s = round(width * r)

    if verb:
        print('Number of pixels to consider: ', s)

    old_time = time.time()

    for y in range(0, height):
        for x in range(0, width):
            values.append(image[y, x])
            mean = 0
            index = 0
            for i in values:
                mean = mean + math.pow((1 - 1 / s), index) * i
                index = + 1

            mean = mean / len(values)

            if image[y, x] < mean * (1 - t / 100):
                res[y, x] = 0
            else:
                res[y, x] = 255

            if verb:
                print('Current pixel value: ', image[y, x])
                print('Array of the latest values: ', values)
                print('Mean of the latest values: ', mean)

            if len(values) == s:
                values.pop(0)
            else:
                continue

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Bradley & Roth Algorithm
################
def brarotMethod(image, dimX, dimY, t=15, verb=False):
    '''
    Bradley & Roth Algorithm
    :param image: image to binarize
    :param r: ratio of pixel to process according to image width
    :param t: percentage do classify the pixel as black or white
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :param verb: True if messages are expected
    :return: returns the binary image
    '''

    height = image.shape[0]
    width = image.shape[1]
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((height - (offY * 2), width - (offX * 2)), dtype=np.uint8)

    old_time = time.time()

    for y in range(offY, height - offY):
        for x in range(offX, width - offX):
            values = [image[y + i][x + j] for i in range(-offY, offY + 1) for j in range(-offX, offX + 1)]
            mean = 0
            for i in values:
                mean = mean + i

            mean = mean / (dimX * dimY)

            if image[y - offY][x - offX] < mean * (1 - t / 100):
                res[y - offY][x - offX] = 0
            else:
                res[y - offY][x - offX] = 255

            if verb:
                print('Current pixel value: ', image[y - offY][x - offX])
                print('Array of the latest values: ', values)
                print('Mean of the latest values: ', mean)

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


def makeMorphKernel(values):
    '''
    Auxilary funtion for morphological operations
    :param values: Values for the kernel
    :return: Kernel/image for operations
    '''

    height = len(values)
    width = len(values[1])
    ker = np.zeros((height, width), dtype=np.uint8)

    for y in range(0, height):
        for x in range(0, width):
            ker[y][x] = values[y][x]

    i = Image.fromarray(ker)
    i.show()
    i = i.convert('L')
    i.save('teste.PNG', 'PNG')

    return


########
# Image dilation
################
def dilation(imIn, ker, verb):
    '''
    Dilation of a binary image
    :param imIn: binary image to dilate
    :param ker: dilation kernel
    :param verb: True if messages are expected
    :return: dilated image
    '''

    (dkY, dkX) = ker.shape
    offX = floor(dkX / 2)
    offY = floor(dkY / 2)
    img = imagePad(imIn, [offY, offY, offX, offX])
    dimY = img.shape[0]
    dimX = img.shape[1]
    imRes = np.zeros((dimY, dimX), dtype=np.uint8)

    old_time = time.time()

    for y in range(offY, dimY - offY):
        for x in range(offX, dimX - offX):
            imRes[y - offY][x - offX] = 0
            for i in range(-offY, offY + 1):
                for j in range(-offX, offX + 1):
                    if img[y + i][x + j] == 1 and ker[i + 1][j + 1] == 1:
                        imRes[y - offY][x - offX] = 1

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return imRes


########
# Image erosion
################
def erosion(imIn, ker, verb):
    '''
    Erodes a binary image with the binary kernel
    :param imIn: binary image to erode
    :param ker: erosion kernel
    :param verb: True if messages are expected
    :return:
    '''

    (dkY, dkX) = ker.shape
    offX = floor(dkX / 2)
    offY = floor(dkY / 2)
    img = imagePad(imIn, [offY, offY, offX, offX])
    dimY = img.shape[0]
    dimX = img.shape[1]
    imRes = np.empty_like(imIn)

    old_time = time.time()

    for y in range(offY, dimY - offY):
        for x in range(offX, dimX - offX):
            imRes[y - offY][x - offX] = 1
            for i in range(-offY, offY + 1):
                for j in range(-offX, offX + 1):
                    if img[y + i][x + j] == 0 and ker[i + 1][j + 1] == 1:
                        imRes[y - offY][x - offX] = 0

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return imRes


########
# Hit and Miss with pad (dont cares are represented as -1 on the kernel)
################
def hitAndMiss(imIn, ker, verb):
    '''
    Hit and Miss operation
    :param imIn: input image
    :param ker: kernel
    :param verb: True if messages are expected
    :return: processed image
    '''

    dkX  = len(ker[1])
    dkY = len(ker)
    offX = floor(dkX / 2)
    offY = floor(dkY / 2)
    img = imagePad(imIn, [offY, offY, offX, offX])
    dimY = img.shape[0]
    dimX = img.shape[1]
    imRes = np.empty_like(imIn)

    old_time = time.time()

    for y in range(offY, dimY - offY):
        for x in range(offX, dimX - offX):
            imRes[y - offY][x - offX] = 1
            for i in range(-offY, offY + 1):
                for j in range(-offX, offX + 1):
                    if ker[i + 1][j + 1] != -1 and ker[i + 1][j + 1] != img[y + i][x + j]:
                        imRes[y - offY][x - offX] = 0

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return imRes


########
# Blum Medial Axis Algorithm
################
def blumAlg(img, verb=False):
    '''
    Skeletization of a binary image
    :param img: input image
    :param verb: True if messages are expected
    :return: returns the skeletization of the input image
    '''

    dimY = img.shape[0]
    dimX = img.shape[1]
    border = np.zeros((dimY, dimX), dtype=np.uint8)
    inside = np.zeros((dimY, dimX), dtype=np.uint8)
    res = np.zeros((dimY, dimX), dtype=np.uint8)
    borderP = []
    insideP = []

    def skeleton(e, f, p):
        minDist = math.sqrt(math.pow((e - p[0][1]), 2) + math.pow((f - p[0][0]), 2))
        indMin = 0
        a = 0
        for pos in p:
            dist = math.sqrt(math.pow((e - pos[1]), 2) + math.pow((f - pos[0]), 2))
            if dist < minDist:
                minDist = dist
                indMin = a
            a += 1

        Np = 1
        a = 0
        if verb:
            print('minDist: ', minDist)
            print('indMin: ', indMin)

        for pos in p:
            if a != indMin:
                dist = math.sqrt(math.pow((e - pos[1]), 2) + math.pow((f - pos[0]), 2))
                if abs(dist - minDist) <= 1 and math.sqrt(
                        math.pow((p[a][1] - p[indMin][1]), 2) + math.pow((p[a][0] - p[indMin][0]), 2)) > 20:
                    Np += 1
            a += 1
        return Np

    def findPos(n, im):
        (c, d) = im.shape
        for z in range(0, c):
            for w in range(0, d):
                if im[z][w] != 0:
                    n.append([z, w])

    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            if img[y][x] == 255:
                img[y][x] = 1

    old_time = time.time()

    # border pixels of the object
    for i in range(1, dimY - 1):
        for j in range(1, dimX - 1):
            if img[i][j] == 1:
                if img[i - 1][j] == 0:
                    border[i][j] = 1
                elif img[i + 1][j] == 0:
                    border[i][j] = 1
                elif img[i][j - 1] == 0:
                    border[i][j] = 1
                elif img[i][j + 1] == 0:
                    border[i][j] = 1

    # interior pixels of the object
    for i in range(0, dimY):
        for j in range(0, dimX):
            inside[i][j] = img[i][j] - border[i][j]

    # get list of border pixel position
    findPos(borderP, border)

    # get list of inside pixel position
    findPos(insideP, inside)

    if verb:
        print('Border:')
        print(border)
        print('Border pixel positions', borderP)
        print('Inside:')
        print(inside)
        print('Inside pixel positions', insideP)

    # compare the distance of each object pixel with the border pixels
    for posObject in insideP:
        if skeleton(posObject[1], posObject[0], borderP) > 1:
            res[posObject[0]][posObject[1]] = 1

    res = res + border

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Zhang-Suen Algorithm
################
def zhSuAlg(img, verb=False, showB=False):
    '''
    Skeletization of a binary image
    :param img: input image
    :param verb: True if messages are expected
    :param showB: Show object border
    :return: returns the skeletization of the input image
    '''

    dimY = img.shape[0]
    dimX = img.shape[1]
    border = np.zeros((dimY, dimX), dtype=np.uint8)
    change = True

    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            if img[y][x] == 255:
                img[y][x] = 1

    if showB:
        for i in range(1, dimY - 1):
            for j in range(1, dimX - 1):
                if img[i][j] == 1:
                    if img[i - 1][j] == 0:
                        border[i][j] = 1
                    elif img[i + 1][j] == 0:
                        border[i][j] = 1
                    elif img[i][j - 1] == 0:
                        border[i][j] = 1
                    elif img[i][j + 1] == 0:
                        border[i][j] = 1

    def countBlack(n):
        s = 0
        for value in n:
            s = s + value

        s = 9 - s

        if verb:
            print('Number of black pixels: ', s)

        return s

    def countTransitions(n):
        c = 0
        for i in range(1, 8):
            if n[i] < n[i + 1]:
                c += 1
        if n[8] < n[1]:
            c += 1

        if verb:
            print('Number of transitions: ', c)

        return c

    old_time = time.time()

    while change:
        change = False
        list1 = []
        list2 = []
        for y in range(1, dimY - 1):
            for x in range(1, dimX - 1):
                values = [img[y][x], img[y - 1][x], img[y - 1][x + 1], img[y][x + 1], img[y + 1][x + 1],
                          img[y + 1][x], img[y + 1][x - 1], img[y][x - 1], img[y - 1][x - 1]]
                if verb:
                    print('Values for step 1: ', values)

                # first step
                if values[0] == 1:
                    if values[1] == 0 or values[3] == 0 or values[5] == 0:
                        if values[3] == 0 or values[5] == 0 or values[7] == 0:
                            if 2 <= countBlack(values) <= 6:
                                if countTransitions(values) == 1:
                                    list1.append([y, x])
                                    change = True

        for pos in list1:
            img[pos[0]][pos[1]] = 0

        for y in range(1, dimY - 1):
            for x in range(1, dimX - 1):
                values = [img[y][x], img[y - 1][x], img[y - 1][x + 1], img[y][x + 1], img[y + 1][x + 1],
                          img[y + 1][x], img[y + 1][x - 1], img[y][x - 1], img[y - 1][x - 1]]

                if verb:
                    print('Values for step 2: ', values)

                # second step
                if values[0] == 1:
                    if values[1] == 0 or values[3] == 0 or values[7] == 0:
                        if values[1] == 0 or values[5] == 0 or values[7] == 0:
                            if 2 <= countBlack(values) <= 6:
                                if countTransitions(values) == 1:
                                    list2.append([y, x])
                                    change = True

        for pos in list2:
            img[pos[0]][pos[1]] = 0

        if change and verb:
            print('A change occurred')
        if not change and verb:
            print('Image done')

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return img + border


########
# Multi class image segmentation
################
def multiSeg(img, verb=False):
    '''
    Segmentation of multiclass images - inspired on the classic algorithm
    :param img: input image with any number of labels
    :param verb: True if messages are expected
    :return: labelled image with a different label for each object and the number of labels on the resulting image
             The result will have labels from 0 sequentially to the number of labels-1
    '''

    def getEqs(cTab, lab):
        for i in range(len(cTab)):
            if lab in cTab[i]:
                return (i)
        return (-1)

    def insertConf(cTab, l1, l2):
        id1 = getEqs(cTab, l1)
        id2 = getEqs(cTab, l2)
        if id1 == -1 and id2 != -1:
            cTab[id2].append(l1)
        elif id1 != -1 and id2 == -1:
            cTab[id1].append(l2)
        elif id1 == -1 and id2 == -1:
            cTab.append([l1, l2])
        else:
            if id1 != id2:
                cTab[id1] += cTab[id2]
                del (cTab[id2])
        return (cTab)

    def labEq(lab, confTab):
        for i in range(len(confTab)):
            if lab in confTab[i]:
                return (min(confTab[i]))
        return (lab)

    height = img.shape[0]
    width = img.shape[1]
    labels = np.empty_like(img, int)

    nextLabel = 0
    labels[0][0] = nextLabel  # First pixel
    nextLabel += 1
    for x in range(1, width):  # First row
        if img[0][x] == img[0][x - 1]:
            labels[0][x] = labels[0][x - 1]
        else:
            labels[0][x] = nextLabel
            nextLabel += 1

    confTab = []

    for y in range(1, height):  # Remaining rows
        if img[y][0] == img[y - 1][0]:  # First column
            labels[y][0] = labels[y - 1][0]
        else:
            labels[y][0] = nextLabel
            nextLabel += 1
        for x in range(1, width):  # Remaining columns
            if img[y][x] == img[y - 1][x]:
                labels[y][x] = labels[y - 1][x]
                if img[y][x] == img[y][x - 1] and labels[y][x] != labels[y][x - 1]:
                    confTab = insertConf(confTab, labels[y][x - 1], labels[y][x])
            elif img[y][x] == img[y][x - 1]:
                labels[y][x] = labels[y][x - 1]
            else:
                labels[y][x] = nextLabel
                nextLabel += 1
    labDict = {}
    off = 0
    for l in range(nextLabel):
        le = labEq(l, confTab)
        if l != le:
            labDict[l] = labDict[le]
            off += 1
        else:
            labDict[l] = l - off
    if verb:
        print('Input image')
        print(img)
        print('Labels before equivalencies')
        print(labels)
        print('Conflit table')
        print(confTab)
        print('Number of labels before equivalencies:', nextLabel)
        print('Dictionary')
        print(labDict)
        print('Number of labels after equivalencies:', nextLabel - off)

    for y in range(0, height):
        for x in range(0, width):
            labels[y][x] = labDict[labels[y][x]]

    if verb:
        print('Labels after equivalencies')
        print(labels)

    return labels, nextLabel - off


########
# Binary image segmentation
################
def binSeg(img, verb=False):
    '''
    Binary segmentation - classic algorithm
    :param img: binary image to segment
    :param verb: True if messages are expected
    :return: segmented image and number of labels (between 1 and N)
    '''

    def getEqs(cTab, lab):
        for i in range(len(cTab)):
            if lab in cTab[i]:
                return (i)
        return (-1)

    def insertConf(cTab, l1, l2):
        id1 = getEqs(cTab, l1)
        id2 = getEqs(cTab, l2)
        if id1 == -1 and id2 != -1:
            cTab[id2].append(l1)
        elif id1 != -1 and id2 == -1:
            cTab[id1].append(l2)
        elif id1 == -1 and id2 == -1:
            cTab.append([l1, l2])
        else:
            if id1 != id2:
                cTab[id1] += cTab[id2]
                del (cTab[id2])
        return (cTab)

    def labEq(lab, confTab):
        for i in range(len(confTab)):
            if lab in confTab[i]:
                return (min(confTab[i]))
        return (lab)

    if verb:
        print('Input image')
        print(img)

    height = img.shape[0]
    width = img.shape[1]

    old_time = time.time()

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
        for x in range(1, width):  # Remaining columns
            if img[y][x] == 1:
                if img[y - 1][x] != 0:
                    img[y][x] = img[y - 1][x]
                    if img[y][x - 1] != 0 and img[y][x - 1] != img[y][x]:
                        if verb:
                            print('Conflict:', img[y][x - 1], img[y][x])
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
    for l in range(1, nextLabel):
        le = labEq(l, confTab)
        if l != le:
            labDict[l] = labDict[le]
            off += 1
        else:
            labDict[l] = l - off
    if verb:
        print('Conflict table')
        print(confTab)
        print('Number of labels before equivalencies:', nextLabel)
        print('Dictionary')
        print(labDict)
        print('Number of labels after equivalencies:', nextLabel - off - 1)

    for y in range(0, height):
        for x in range(0, width):
            img[y][x] = labDict[img[y][x]]

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    if verb:
        print('Labels after equivalencies')
        print(img)

    return img, nextLabel - off - 1


########
# Binary image segmentation
################
def binItSeg(img, verb=False):
    '''
    Binary segmentation - iterative algorithm
    :param img: binary image to segment
    :param verb: True if messages are expected
    :return: segmented image and number of labels (between 1 and N)
    '''

    height = img.shape[0]
    width = img.shape[1]
    change = True
    cnt = 0
    res = np.zeros((height, width), dtype=np.long)
    labels = []
    color = 0

    old_time = time.time()

    # Labels between 1 and N
    nextLabel = 1
    if img[0][0] == 1:
        res[0][0] = nextLabel  # First pixel
        nextLabel += 1

    for x in range(1, width):
        if img[0][x] == 1:
            res[0][x] = nextLabel
            nextLabel += 1

    for y in range(1, height):  # Remaining rows
        if img[y][0] == 1:
            res[y][0] = nextLabel
            nextLabel += 1
        for x in range(1, width):  # Remaining columns
            if img[y][x] == 1:
                res[y][x] = nextLabel
                nextLabel += 1

    if verb:
        print('First labels')
        print(res)

    # Zig-Zag through the image
    while change:
        change = False
        cnt = cnt + 1
        for x in range(1, width):  # First row
            if res[0][x] != 0:
                if res[0][x - 1] < res[0][x] and res[0][x - 1] != 0:
                    res[0][x] = res[0][x - 1]
                    change = True
                if res[1][x] < res[0][x] and res[1][x] != 0:
                    res[0][x] = res[1][x]
                    change = True

        for y in range(1, height):  # Remaining rows
            if res[y][0] != 0:
                if res[y - 1][0] < res[y][0] and res[y - 1][0] != 0:
                    res[y][0] = res[y - 1][0]
                    change = True
                elif res[y + 1][0] < res[y][0] and res[y + 1][0] != 0:
                    res[y][0] = res[y + 1][0]
                    change = True
                elif res[y][1] < res[y][0] and res[y][1] != 0:
                    res[y][0] = res[y][1]
                    change = True
            for x in range(1, width):  # Remaining columns
                if res[y][x] != 0:
                    if res[y - 1][x] < res[y][x] and res[y - 1][x] != 0:
                        res[y][x] = res[y - 1][x]
                        change = True
                    elif res[y][x - 1] < res[y][x] and res[y][x - 1] != 0:
                        res[y][x] = res[y][x - 1]
                        change = True
                    if x != width - 1:
                        if res[y][x + 1] < res[y][x] and res[y][x + 1] != 0:
                            res[y][x] = res[y][x + 1]
                            change = True
                    if y != height - 1:
                        if res[y + 1][x] < res[y][x] and res[y + 1][x] != 0:
                            res[y][x] = res[y + 1][x]
                            change = True
        if verb:
            print('Current labels')
            print(res)

        if change:
            change = False
            cnt = cnt + 1
            for x in range(width - 1, 0, -1):  # Last row
                if res[height - 1][x] != 0:
                    if res[height - 1][x - 1] < res[height - 1][x] and res[height - 1][x - 1] != 0:
                        res[height - 1][x] = res[height - 1][x - 1]
                        change = True

            for y in range(height - 1, 0, -1):  # Remaining rows
                if res[y][width - 1] != 0:
                    if y != 0:
                        if res[y - 1][width - 1] < res[y][width - 1] and res[y - 1][width - 1] != 0:
                            res[y][width - 1] = res[y - 1][width - 1]
                            change = True
                        elif res[y - 1][width - 2] < res[y][width - 1] and res[y - 1][width - 2] != 0:
                            res[y][width - 1] = res[y - 1][width - 2]
                            change = True
                    if y != height - 1:
                        if res[y + 1][width - 1] < res[y][width - 1] and res[y + 1][width - 1] != 0:
                            res[y][width - 1] = res[y + 1][width - 1]
                            change = True
                for x in range(width - 1, 0, -1):  # Remaining columns
                    if res[y][x] != 0:
                        if res[y - 1][x] < res[y][x] and res[y - 1][x] != 0:
                            res[y][x] = res[y - 1][x]
                            change = True
                        elif res[y][x - 1] < res[y][x] and res[y][x - 1] != 0:
                            res[y][x] = res[y][x - 1]
                            change = True
                        if x != width - 1:
                            if res[y][x + 1] < res[y][x] and res[y][x + 1] != 0:
                                res[y][x] = res[y][x + 1]
                                change = True
                        if y != height - 1:
                            if res[y + 1][x] < res[y][x] and res[y + 1][x] != 0:
                                res[y][x] = res[y + 1][x]
                                change = True

            if verb:
                print('Current labels')
                print(res)

    for i in range(0, height):
        for j in range(0, width):
            if not (labels.__contains__(res[i][j])) and res[i][j] != 0:
                labels.append(res[i][j])

    if verb:
        print("Object labels: ", labels)

    color = int(255 / len(labels))

    for i in range(0, height):
        for j in range(0, width):
            if labels.__contains__(res[i][j]):
                res[i][j] = color * (labels.index(res[i][j]) + 1)

    img = res

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    if verb:
        print("The algorithm ran ", cnt, " times.")

    return img


########
# Horizontal Vertical Diagonal Projections
################
def imgProjXYD(img, X=False, Y=False, D1=False, D2=False):
    '''
    Image Projections
    :param img: binary image
    :param X: True if Horizontal Projection
    :param Y: True if Vertical Projection
    :param D1: True if Diagonal Projection (45º)
    :param D2: True if Diagonal Projection (135º)
    :return:
    '''

    height = img.shape[0]
    width = img.shape[1]
    sum_y = []
    sum_x = []
    sum_d1 = []
    sum_d2 = []

    if Y:
        for i in range(0, height):
            cnt = 0
            for j in range(0, width):
                if img[i, j] == 1:
                    cnt = cnt + 1

            sum_x.append([i, cnt])

        y = []
        x = []
        for entry in sum_x:
            y.append(entry[1])
            x.append(entry[0])

        plt.plot(x, y)
        plt.yticks(y)
        plt.xticks(x)
        plt.xlabel("Position")
        plt.ylabel("Number of Pixels")
        plt.title("Vertical Projection")
        plt.show()

    if X:
        for i in range(0, width):
            cnt = 0
            for j in range(0, height):
                if img[j, i] == 1:
                    cnt = cnt + 1

            sum_y.append([i, cnt])

        y = []
        x = []
        for entry in sum_y:
            y.append(entry[1])
            x.append(entry[0])

        plt.plot(x, y)
        plt.yticks(y)
        plt.xticks(x)
        plt.xlabel("Position")
        plt.ylabel("Number of Pixels")
        plt.title("Horizontal Projection")
        plt.show()

    if D1:
        if img[0, 0] == 1:
            sum_d1.append([0, 1])
        else:
            sum_d1.append([0, 0])

        dif = 0
        for i in range(1, height + width - 1):
            y = i
            x = 0
            cnt = 0
            if i > height - 1:
                y = height - 1
                dif = dif + 1
                x = x + dif
            for j in range(0, i - dif * 2 + 1):
                if img[y, x] == 1:
                    cnt = cnt + 1
                y = y - 1
                x = x + 1

            sum_d1.append([i, cnt])

        y = []
        x = []
        for entry in sum_d1:
            y.append(entry[1])
            x.append(entry[0])

        plt.plot(x, y)
        plt.yticks(y)
        plt.xticks(x)
        plt.xlabel("Position")
        plt.ylabel("Number of Pixels")
        plt.title("Diagonal Projection (45º)")
        plt.show()

    if D2:
        if img[0, width - 1] == 1:
            sum_d2.append([0, 1])
        else:
            sum_d2.append([0, 0])

        dif = 0
        for i in range(1, height + width - 1):
            y = i
            x = width - 1
            cnt = 0
            if i > height - 1:
                y = height - 1
                dif = dif + 1
                x = x - dif
            for j in range(0, i - 2 * dif + 1):
                if img[y, x] == 1:
                    cnt = cnt + 1
                y = y - 1
                x = x - 1

            sum_d2.append([i, cnt])

        y = []
        x = []
        for entry in sum_d2:
            y.append(entry[1])
            x.append(entry[0])

        plt.plot(x, y)
        plt.yticks(y)
        plt.xticks(x)
        plt.xlabel("Position")
        plt.ylabel("Number of Pixels")
        plt.title("Diagonal Projection (135º)")
        plt.show()


########
# Auxiliary to Region Growing
################
def getValueList(img, values):
    '''
    Get positions for values
    :param img: grayscale image
    :param values: pixel intensities to search ([])
    :return: list of positions for values
    '''

    height = img.shape[0]
    width = img.shape[1]
    pos = []

    for i in range(0, height):
        for j in range(0, width):
            for z in range(0, len(values)):
                if img[i, j] == values[z]:
                    values.pop(z)
                    pos.append([i, j])

    return pos


########
# Region Growing
################
def regGro(img, seeds, t, sel=False, verb=False):
    '''
    Region Growing
    :param img: grayscale image
    :param seeds: seed points ([y,x])
    :param t: threshold for pixel acceptance
    :param sel: select 4 (False) or 8 (True) connected neighbourhood
    :param verb: True if messages expected
    :return: returns segmented image
    '''

    height = img.shape[0]
    width = img.shape[1]
    res = np.zeros((height, width), dtype=np.uint8)
    mark = 1

    def selectNeighbourhood(s):
        if s:
            n = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        else:
            n = [[0, -1], [1, 0], [0, 1], [-1, 0]]

        return n

    neigh = selectNeighbourhood(sel)

    if verb:
        print(len(neigh), "-connected neighbourhood")

    old_time = time.time()

    while len(seeds) > 0:
        currentSeed = seeds.pop(0)
        if verb:
            print("Current seed: ", currentSeed)

        res[currentSeed[0], currentSeed[1]] = mark

        for pos in neigh:
            x = currentSeed[1] + pos[1]
            y = currentSeed[0] + pos[0]

            if x < 0 or y < 0 or x >= width or y >= height:
                continue

            diff = abs(img[currentSeed[0], currentSeed[1]] - img[y, x])
            if verb:
                print("Difference with neighbourhood [", y, x, "]: ", diff)

            if diff < t and res[y, x] == 0:
                res[y, x] = mark
                seeds.append([y, x])
        if verb:
            print("")

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Object Characteristics Extraction
################
def objCharact(img, sel=False, verb=False):
    '''
    Gets/Calculates object properties
    :param img: binary image with a single object
    :param sel: select 4 (False) or 8 (True) connected neighbourhood
    :param verb: True if messages are expected
    :return: list of object properties
    '''

    height = img.shape[0]
    width = img.shape[1]
    cntP = 0
    cntA = 0
    stop = False
    last_x = -1
    last_y = -1
    polarCoord = []
    imgcpy = np.zeros((height, width), dtype=np.uint8)

    # Object Area
    for i in range(0, height):
        for j in range(0, width):
            if img[i, j] == 1:
                cntA = cntA + 1

    print("Area: ", cntA)

    ###############

    # Object Centroid
    cntI = 0
    cntJ = 0
    cntII = 0
    cntJJ = 0
    cntIJ = 0

    for i in range(0, height):
        for j in range(0, width):
            if img[i, j] == 1:
                cntI = cntI + i
                cntJ = cntJ + j
                cntII = cntII + math.pow(i, 2)
                cntJJ = cntJJ + math.pow(j, 2)
                cntIJ = cntIJ + i * j

    cY = cntI / cntA
    cX = cntJ / cntA

    if verb:
        print("Object Centroid (True Coordinates): [", cY, ",", cX, "]")

    cY = round(cY)
    cX = round(cX)

    print("Object Centroid: [", cY, ",", cX, "]")

    ###############

    # Object Perimeter
    def selectNeighbourhood(s):
        if s:
            n = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]
        else:
            n = [[0, 1], [1, 0], [0, -1], [-1, 0]]

        return n

    def nextDir8(argument):
        switcher = {
            0: 5,
            1: 6,
            2: 7,
            3: 0,
            4: 1,
            5: 2,
            6: 3,
            7: 4,
        }
        return switcher.get(argument, "-1")

    def nextDir4(argument):
        switcher = {
            0: 3,
            1: 0,
            2: 1,
            3: 2,
        }
        return switcher.get(argument, "-1")

    neigh = selectNeighbourhood(sel)

    for i in range(0, height):
        if stop:
            break
        for j in range(0, width):
            if img[i, j] == 1:
                initial_y = i
                initial_x = j
                stop = True
                break

    curr_x = initial_x
    curr_y = initial_y

    if verb and sel:
        print("Selected neighbourhood 8")

    if verb and not sel:
        print("Selected neighbourhood 4")

    if verb:
        print("\nCoordinates for border points\n----")

    for p in range(0, len(neigh)):
        if img[curr_y + neigh[p][0], curr_x + neigh[p][1]] == 1:
            if curr_y + neigh[p][0] == last_y and curr_x + neigh[p][1] == last_x:
                continue
            else:
                initial_dir = p
                last_x = curr_x
                last_y = curr_y
                curr_y = curr_y + neigh[p][0]
                curr_x = curr_x + neigh[p][1]

                y = curr_y - cY
                x = curr_x - cX
                polarCoord.append([math.sqrt(math.pow(y, 2) + math.pow(x, 2)), [curr_y, curr_x]])

                if verb:
                    print(last_y, last_x, p)

                break

    while stop:
        if sel:
            p = nextDir8(p)
        else:
            p = nextDir4(p)

        while True:
            if img[curr_y + neigh[p][0], curr_x + neigh[p][1]] == 1:
                curr_dir = p
                last_x = curr_x
                last_y = curr_y
                curr_y = curr_y + neigh[p][0]
                curr_x = curr_x + neigh[p][1]

                y = curr_y - cY
                x = curr_x - cX
                polarCoord.append([math.sqrt(math.pow(y, 2) + math.pow(x, 2)), [curr_y, curr_x]])

                if verb:
                    print(last_y, last_x, p)

                if sel:
                    if p == 0 or p == 2 or p == 4 or p == 6:
                        cntP = cntP + 1
                    else:
                        cntP = cntP + math.sqrt(2)
                else:
                    cntP = cntP + 1

                break

            p += 1
            if p > 7 and sel:
                p = 0
            if p > 3 and not sel:
                p = 0

        if last_x == initial_x and last_y == initial_y and curr_dir == initial_dir:
            stop = False

    if verb:
        print("----")

    print("Perimeter: ", cntP)

    ###############

    # Object Form Factor
    ff = (4 * math.pi * cntA) / math.pow(cntP, 2)

    print("Form factor: ", ff)

    ###############

    for i in range(0, height):
        if stop:
            break
        for j in range(0, width):
            if img[i, j] == 1:
                upperX = j
                upperY = i
                stop = True
                break

    if verb:
        print("Upper pixel coordinates: [", upperY, ",", upperX, "]")

    stop = False

    for i in range(height - 1, 0, -1):
        if stop:
            break
        for j in range(width - 1, 0, -1):
            if img[i, j] == 1:
                lowerX = j
                lowerY = i
                stop = True
                break

    if verb:
        print("Lower pixel coordinates: [", lowerY, ",", lowerX, "]")

    stop = False

    for i in range(width - 1, 0, -1):
        if stop:
            break
        for j in range(height - 1, 0, -1):
            if img[j, i] == 1:
                rightX = i
                rightY = j
                stop = True
                break

    if verb:
        print("Right pixel coordinates: [", rightY, ",", rightX, "]")

    stop = False

    for i in range(0, width):
        if stop:
            break
        for j in range(0, height):
            if img[j, i] == 1:
                leftX = i
                leftY = j
                stop = True
                break

    if verb:
        print("Left pixel coordinates: [", leftY, ",", leftX, "]")

    # Get Max Diameter
    if rightX - leftX < lowerX - upperX:
        w = abs(lowerX - upperX) + 1
        h = abs(lowerY - upperY) + 1
        mxD = math.sqrt(math.pow(h, 2) + math.pow(w, 2))
        if verb:
            print("Height: ", h, "\nWidth: ", w)
    else:
        w = abs(rightX - leftX) + 1
        h = abs(rightY - leftY) + 1
        mxD = math.sqrt(math.pow(h, 2) + math.pow(w, 2))
        if verb:
            print("Height: ", h, "\nWidth: ", w)

    print("Max Diameter: ", mxD)

    ###############

    # Get Compactness
    cmp = (4 * math.pi * cntA)/math.pow(cntP, 2)
    cmp = math.pow(cntP, 2)/cntA

    print("Compactness: ", cmp)

    cmpN = 1 - 4 * math.pi / cmp
    if cmpN < 0:
        cmpN = 1 + cmpN

    print("Normalized compactness: ", cmpN)

    ###############

    # Object Orientation
    mII = cntII - math.pow(cntI, 2) / cntA
    mJJ = cntJJ - math.pow(cntJ, 2) / cntA
    mIJ = cntIJ - (cntI * cntJ) / cntA
    if mIJ != 0:
        ori = math.atan((mII - mJJ + math.sqrt(math.pow((mII - mJJ), 2) + 4 * math.pow(mIJ, 2))) / (2 * mIJ))
        ori = -ori * 180 / math.pi
    else:
        ori = '?'
        print("Cannot obtain object orientation.")

    if verb:
        print("Sx: ", cntI, "\nSy: ", cntJ, "\nSxx: ", cntII, "\nSyy: ", cntJJ, "\nSxy: ", cntIJ, "\nMxx: ", mII,
              "\nMyy: ", mJJ, "\nMxy: ", mIJ)
        print("Object Orientation: ", ori, "º")

    ###############

    # Convex Perimeter

    minD = []
    maxD = []
    convexP = 0

    if polarCoord[0][0] > polarCoord[1][0] < polarCoord[2][0]:
        look = 'max'
        minD.append(polarCoord[1][1])
    elif polarCoord[0][0] < polarCoord[1][0] < polarCoord[2][0]:
        look = 'max'

    if polarCoord[0][0] < polarCoord[1][0] > polarCoord[2][0]:
        look = 'min'
        maxD.append(polarCoord[1][1])
    elif polarCoord[0][0] > polarCoord[1][0] > polarCoord[2][0]:
        look = 'min'

    for i in range(3, len(polarCoord)):
        if look == 'min':
            if polarCoord[i-1][0] < polarCoord[i][0]:
                look = 'max'
                minD.append(polarCoord[i-1][1])
        else:
            if polarCoord[i - 1][0] > polarCoord[i][0]:
                look = 'min'
                maxD.append(polarCoord[i-1][1])

    for i in range(0, len(maxD)-1):
        convexP = convexP + math.sqrt(math.pow(maxD[i][0] - maxD[i+1][0], 2) + math.pow(maxD[i][1] - maxD[i+1][1], 2))

    convexP = convexP + math.sqrt(math.pow(maxD[len(maxD)-1][0] - maxD[0][0], 2) + math.pow(maxD[len(maxD)-1][1] - maxD[0][1], 2))

    if verb:
        print('Distance to centroid: ', polarCoord)
        print('Min pixels: ', minD)
        print('Max pixels: ', maxD)

    print('Convex Perimeter: ', convexP)

    ###############

    # Convex Area

    convexA = 0
    end = True
    convexBorder = []

    if maxD[0][1] - maxD[len(maxD) - 1][1] != 0:
        if maxD[0][0] - maxD[len(maxD) - 1][0] != 0:
            m = (maxD[len(maxD) - 1][0] - maxD[0][0]) / (maxD[len(maxD) - 1][1] - maxD[0][1])
            numberX = abs(maxD[0][1] - maxD[len(maxD) - 1][1])
            b = maxD[i]

            if m > 0 and maxD[len(maxD) - 1][0] > maxD[0][0]:
                varY = 1
                varX = 1
            elif m < 0 and maxD[len(maxD) - 1][0] > maxD[0][0]:
                varY = 1
                varX = -1
                m = m * -1
            elif m < 0 and maxD[len(maxD) - 1][0] < maxD[0][0]:
                varY = -1
                varX = 1
                m = m * -1
            elif m > 0 and maxD[len(maxD) - 1][0] < maxD[0][0]:
                varY = -1
                varX = -1

            for j in range(1, numberX + 1):
                valueY0 = round(j * m * varY + b[0])
                valueY1 = round(j * m * varX + b[1])
                if verb:
                    print('Convex shape new pixels [y,x]: ', '[', valueY0, ']', '[', valueY1, ']')
                imgcpy[valueY0][valueY1] = 1
                convexBorder.append([valueY0, valueY1])
        else:
            if maxD[0][1] > maxD[len(maxD) - 1][1]:
                a = maxD[len(maxD) - 1][1]
                while maxD[0][1] != a - 1:
                    if verb:
                        print('Convex shape new pixels [y,x]: ', '[', maxD[0][0], ']', '[', a, ']')
                    imgcpy[maxD[0][0]][a] = 1
                    convexBorder.append([maxD[0][0], a])
                    a += 1
            elif maxD[0][1] < maxD[len(maxD) - 1][1]:
                a = maxD[0][1]
                while maxD[len(maxD) - 1][1] != a:
                    if verb:
                        print('Convex shape new pixels [y,x]: ', '[', maxD[0][0], ']', '[', a, ']')
                    imgcpy[maxD[0][0]][a] = 1
                    convexBorder.append([maxD[0][0], a])
                    a += 1
    else:
        if maxD[0][0] > maxD[len(maxD) - 1][0]:
            a = maxD[0][0]
            while maxD[len(maxD) - 1][0] != a - 1:
                a -= 1
                imgcpy[a][maxD[0][1]] = 1
                convexBorder.append([a, maxD[0][1]])
                if verb:
                    print('Convex shape new pixels [y,x]: ', '[', a, ']', '[', maxD[0][1], ']')
        elif maxD[0][0] < maxD[len(maxD) - 1][0]:
            a = maxD[0][0]
            while maxD[len(maxD) - 1][0] != a + 1:
                a += 1
                imgcpy[a][maxD[0][1]] = 1
                convexBorder.append([a, maxD[0][1]])
                if verb:
                    print('Convex shape new pixels [y,x]: ', '[', a, ']', '[', maxD[0][1], ']')

    for i in range(0, len(maxD)-1):
        if maxD[i][1] - maxD[i + 1][1] != 0:
            if maxD[i][0] - maxD[i+1][0] != 0:
                m = (maxD[i+1][0] - maxD[i][0])/(maxD[i+1][1] - maxD[i][1])
                numberX = abs(maxD[i][1] - maxD[i + 1][1])
                b = maxD[i]

                if m > 0 and maxD[i + 1][0] > maxD[i][0]:
                    varY = 1
                    varX = 1
                elif m < 0 and maxD[i + 1][0] > maxD[i][0]:
                    varY = 1
                    varX = -1
                    m = m * -1
                elif m < 0 and maxD[i + 1][0] < maxD[i][0]:
                    varY = -1
                    varX = 1
                    m = m * -1
                elif m > 0 and maxD[i + 1][0] < maxD[i][0]:
                    varY = -1
                    varX = -1

                for j in range(1, numberX+1):
                    valueY0 = round(j * m * varY + b[0])
                    valueY1 = round(j * m * varX + b[1])
                    if verb:
                        print('Convex shape new pixels [y,x]: ', '[', valueY0, ']', '[', valueY1, ']')
                    imgcpy[valueY0][valueY1] = 1
                    convexBorder.append([valueY0, valueY1])
            else:
                if maxD[i][1] > maxD[i + 1][1]:
                    a = maxD[i][1]
                    while maxD[i + 1][1] != a + 1:
                        if verb:
                            print('Convex shape new pixels [y,x]: ', '[', maxD[i][0], ']', '[', a, ']')
                        imgcpy[maxD[i][0]][a] = 1
                        convexBorder.append([maxD[i][0], a])
                        a -= 1
                elif maxD[i][1] < maxD[i + 1][1]:
                    a = maxD[i][1]
                    while maxD[i + 1][1] != a:
                        a += 1
                        if verb:
                            print('Convex shape new pixels [y,x]: ', '[', maxD[i][0], ']', '[', a, ']')
                        imgcpy[maxD[i][0]][a] = 1
                        convexBorder.append([maxD[i][0], a])

        else:
            if maxD[i][0] > maxD[i + 1][0]:
                a = maxD[i][0]
                while maxD[i + 1][0] != a:
                    a -= 1
                    imgcpy[a][maxD[i][1]] = 1
                    convexBorder.append([a, maxD[i][1]])
                    if verb:
                        print('Convex shape new pixels [y,x]: ', '[', a, ']', '[', maxD[i][1], ']')
            elif maxD[i][0] < maxD[i + 1][0]:
                a = maxD[i][0]
                while maxD[i+1][0] != a:
                    a += 1
                    imgcpy[a][maxD[i][1]] = 1
                    convexBorder.append([a, maxD[i][1]])
                    if verb:
                        print('Convex shape new pixels [y,x]: ', '[', a, ']', '[', maxD[i][1], ']')

    if verb:
        print('Convex border:')
        print(imgcpy)
        print('Convex border positions: ', convexBorder)

    for i in range(0, len(convexBorder)-2):
        convexA += convexBorder[i][1] * convexBorder[i + 1][0]
        convexA -= convexBorder[i + 1][1] * convexBorder[i][0]

    convexA += convexBorder[len(convexBorder)-1][1] * convexBorder[0][0]
    convexA -= convexBorder[0][1] * convexBorder[len(convexBorder)-1][0]

    convexA /= 2

    print('Convex Area: ', convexA)

    ###############

    # Get Convexity
    cov = convexP/cntP

    print("Convexity: ", cov)

    ###############

    # Get Solidity
    if convexA != 0:
        sd = cntA/convexA
        print("Solidity: ", sd)

    ###############

    # Get Circularity
    cir = (4 * math.pi * cntA) / math.pow(convexP, 2)

    print("Circularity: ", cir)

    ###############
