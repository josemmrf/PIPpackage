from statistics import median
from math import sin, cos
from copy import deepcopy
import random
import numpy as np
import time
from math import floor, ceil


# ADAPTED FUNCTIONS

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
    (dimY, dimX) = img.shape
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
    return x * s, y * s


########
# Calculate translation destiny coordinates
################
def calcZoomCoord(x, y, s):
    '''
    Zoom Operation
    :param x: x value position
    :param y: y value position
    :param s: scaling factor
    :return: new coordinates
    '''
    return x / s, y / s


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
    res = a + (b - a) * off
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
    return int(round(res, 0))


########
# Bicubic interpolation Aux
################
def bicubic_aux(values, off, verb=False):
    '''
    Bilcubic interpolation Auxiliary Function
    :param v1: Fist value
    :param v2: Second value
    :param v3: Third value
    :param v4: Fourth value
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
    Bicubic interpolation
    :param xorig: X coordinate
    :param yorig: Y coordinate
    :param img: Image to interpolate
    :param verb: True if messages are expected
    :return: rounded value of resulting bicubic interpolation
    '''
    if verb:
        print('Bicubic to find', xorig, yorig)

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

    return int(round(res, 0))


########
# Mean Filter Type A
################
def meanFilterA(image, dimX, dimY, verb=False):
    '''
    Mean filter
    :param image: image to be filtered
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :return: returns the result image
    '''

    count_sum = 0
    count_div = 0
    (height, width) = image.shape
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((width - (offX * 2), height - (offY * 2)), dtype=int)

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
    :return: returns the result image
    '''

    count_div = 0
    (height, width) = image.shape
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((width - (offX * 2), height - (offY * 2)), dtype=int)

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
            res[y][x] = res[y][x] / (dimX * dimY)
            count_div += 1

    new_time = time.time()

    if verb:
        print('Number of sums: ', count_sum)
        print('Number of div: ', count_div)

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Mean Filter Type C
################
def meanFilterC(image, dimX, dimY, verb=False):
    '''
    Mean filter
    :param image: image to be filtered
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :return: returns the result image
    '''

    count_div = 0
    (height, width) = image.shape
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    conneroff = [[-(offY + 1), -(offX + 1)], [-(offY + 1), offX], [offY, -(offX + 1)], [offY, offX]]
    res = np.zeros((width - (offX * 2), height - (offY * 2)), dtype=int)

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
            res[y][x] = res[y][x] / (dimX * dimY)
            count_div += 1

    new_time = time.time()

    if verb:
        print('Number of sums: ', count_sum)
        print('Number of div: ', count_div)

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


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
    :return: returns the result image
    '''

    (height, width) = image.shape
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((width - (offX * 2), height - (offY * 2)), dtype=int)

    old_time = time.time()

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
            res[y - offY][x - offX] = mean / (dimX * dimY)
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
    :return: returns the result image
    '''

    (height, width) = image.shape
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((width - (offX * 2), height - (offY * 2)), dtype=int)

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
    res = np.zeros(len(unsorted), dtype=int)

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
    res = np.zeros(int((dimX * dimY) / 2 + 1), dtype=int)

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
def medianFilter(image, dimX, dimY, verb=False):
    '''
    Median filter
    :param image: image to be filtered
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :return: returns the result image
    '''

    (height, width) = image.shape
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((width - (offX * 2), height - (offY * 2)), dtype=int)

    old_time = time.time()

    for y in range(offY, height - offY):
        for x in range(offX, width - offX):
            values = [image[y + i][x + j] for i in range(-offY, offY + 1) for j in range(-offX, offX + 1)]

            if verb:
                print('Values to consider:', values)
            sorted_values = bubblesortA(values, 'A')
            if verb:
                print('Sorted values:', sorted_values)
                print('Median value: ', sorted_values[int((dimX * dimY) / 2)])

            res[y - offY][x - offX] = sorted_values[int((dimX * dimY) / 2)]

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Mode Filter
################
def modeFilter(image, dimX, dimY, verb=False):
    '''
    Mode filter
    :param image: image to be filtered
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :return: returns the result image
    '''

    def exists(p, tab):
        for j in range(0, len(tab)):
            if p in tab[j]:
                return 0
        return -1

    (height, width) = image.shape
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((width - (offX * 2), height - (offY * 2)), dtype=int)

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
                if count < entry[1]:
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
    :return: returns the result image
    '''

    listoff = [[0, 0], [0, 1], [1, 0]]
    (height, width) = image.shape
    res = np.zeros((width, height), dtype=int)

    old_time = time.time()

    for y in range(0, height):
        for x in range(0, width):
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
    :return: returns the result image
    '''

    listoff = [[0, 0], [0, 1], [1, 0], [1, 1]]
    (height, width) = image.shape
    res = np.zeros((width, height), dtype=int)

    old_time = time.time()

    for y in range(0, height):
        for x in range(0, width):
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
# Sobel Filter
################
def sobelFilter(image, verb=False):
    '''
    Sobel filter
    :param image: image to be filtered
    :return: returns the result image
    '''

    listoffx = [[-1, -1], [1, -1], [-1, 1], [1, 1], [0, -1], [0, 1]]
    listoffy = [[-1, -1], [-1, 1], [-1, 0], [1, -1], [1, 1], [1, 0]]
    (height, width) = image.shape
    res = np.zeros((width, height), dtype=int)

    old_time = time.time()

    for y in range(1, height + 1):
        for x in range(1, width + 1):
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
    :return: returns the result image
    '''

    (height, width) = image.shape
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((width - (offX * 2), height - (offY * 2)), dtype=int)

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

            for value in diffs:
                mean += value

            res[y - offY][x - offX] = mean / k

    new_time = time.time()

    if verb:
        print('Execution time (seconds): ', format(new_time - old_time, '.3f'))

    return res


########
# Sigma Filter
################
def sigmaFilter(image, dimX, dimY, s, verb=False):
    '''
    K-Nearest Neighbour Filter
    :param image: image to be filtered
    :param dimX: X dimension of the kernel
    :param dimY: Y dimension of the kernel
    :param s: standard variation
    :return: returns the result image
    '''

    (height, width) = image.shape
    offX = int(dimX / 2)
    offY = int(dimY / 2)
    res = np.zeros((width - (offX * 2), height - (offY * 2)), dtype=int)

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
    (height, width) = image.shape
    res = image
    var = 0
    for x in range(0, width):
        for y in range(0, height):
            var = var + image[x][y]
            res[x][y] = var


########
# Hybrid 5x5 median filter value for a pixel
################
def medianHybrid5x5(x, y, image, verb=False):
    '''
    Hybrid median filter
    :param x: X coordinate
    :param y: Y coordinate
    :param image: image to be filtered
    :return:
    '''
    diag = [[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2], [2, -2], [1, -1], [-1, 1], [-2, 2]]
    cross = [[0, -2], [0, -1], [0, 0], [0, 1], [0, 2], [-2, 0], [-1, 0], [1, 0], [2, 0]]

    def getValues(x, y, off, img):
        l = np.empty([len(off)])
        for i in range(len(off)):
            l[i] = img[y + off[i][1]][x + off[i][0]]
        return (l)

    img = imagePad(image, [2, 2, 2, 2])
    if verb:
        print('Imagem com duplicacao das margens(2)\n')
        printImg(img)
        print()
    print('Pixel x=', x, 'y=', y)
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
    for j in range(len(imgRes)):
        for i in range(len(imgRes[j])):
            imgRes[i][j] = newLevels[imgRes[i][j]]
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
def dilation(imIn, ker):
    '''
    Dilation of a binary image
    :param imIn: binary image to dilate
    :param ker: dilation kernel
    :return: dilated image
    '''
    (dkY, dkX) = ker.shape
    offX = floor(dkX / 2)
    offY = floor(dkY / 2)
    img = imagePad(imIn, [offY, offY, offX, offX])
    (dimY, dimX) = img.shape
    imRes = np.empty_like(imIn)

    for y in range(offY, dimY - offY):
        for x in range(offX, dimX - offX):
            imRes[y - offY][x - offX] = 0
            for i in range(-offY, offY + 1):
                for j in range(-offX, offX + 1):
                    if img[y + i][x + j] == 1 and ker[i + 1][j + 1] == 1:
                        imRes[y - offY][x - offX] = 1
    return imRes


########
# Image erosion
################
def erosion(imIn, ker):
    '''
    Erodes a binary image with the binary kernel ker
    :param imIn:
    :param ker:
    :return:
    '''
    (dkY, dkX) = ker.shape
    offX = floor(dkX / 2)
    offY = floor(dkY / 2)
    img = imagePad(imIn, [offY, offY, offX, offX])
    (dimY, dimX) = img.shape
    imRes = np.empty_like(imIn)

    for y in range(offY, dimY - offY):
        for x in range(offX, dimX - offX):
            imRes[y - offY][x - offX] = 1
            for i in range(-offY, offY + 1):
                for j in range(-offX, offX + 1):
                    if img[y + i][x + j] == 0 and ker[i + 1][j + 1] == 1:
                        imRes[y - offY][x - offX] = 0
    return imRes


########
# Hit and Miss with pad (dont cares are represented as -1 on the kernel)
################
def hitAndMiss(imIn, ker):
    '''
    Hit and Miss operation
    :param imIn: input image
    :param ker: kernel
    :return: processed image
    '''

    (dkY, dkX) = ker.shape
    offX = floor(dkX / 2)
    offY = floor(dkY / 2)
    img = imagePad(imIn, [offY, offY, offX, offX])
    (dimY, dimX) = img.shape
    imRes = np.empty_like(imIn)

    for y in range(offY, dimY - offY):
        for x in range(offX, dimX - offX):
            imRes[y - offY][x - offX] = 1
            for i in range(-offY, offY + 1):
                for j in range(-offX, offX + 1):
                    if ker[i + 1][j + 1] != -1 and ker[i + 1][j + 1] != img[y + i][x + j]:
                        imRes[y - offY][x - offX] = 0
    return imRes


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

    (height, width) = img.shape
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

    return (labels, nextLabel - off)


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

    if verb:
        print('Labels after equivalencies')
        print(img)

    return (img, nextLabel - off - 1)
