import PIPpackage
#import numpy as np

# #######
# # MAIN PROGRAM
# #######################
#
# #insc = pd.read_excel(os.path.join(filesPath,'inscritosSS.xlsx'),engine='openpyxl')
# #makeFolders()
# #for i in range(len(insc)):
# #    genExam(insc['Numero'][i],insc['Nome'][i])
#
# mail = loginEmail('ssfctunl@gmail.com', 'Sensoriais2020')  # 'simfctunl@gmail.com', 'sim576911')
# checkEmail(mail)

# im=PIPpackage.genImage(7,7,8)
# print(im)
# print()
# imD=PIPpackage.imagePad(im,[2,2,2,2],mode='wrap')
# PIPpackage.printImg(imD)

#######################################
img = PIPpackage.genImage(4, 4, 10, 10)
PIPpackage.printImg(img)
# print('Bilinear: ', PIPpackage.bilinear(5.5, 4.5, img))
# print('Bicubic: ', PIPpackage.PIPpackage.bicubic(5.5, 4.5, img))

# print('Result', PIPpackage.PIPpackage.sorting([5,4,8,2,7,1,6], 'D'))

# print('Result: ', PIPpackage.PIPpackage.medianFilter3x3(img, True))
img = PIPpackage.imagePad(img, [2, 2, 2, 2])
PIPpackage.printImg(img)
print('Result: ', PIPpackage.PIPpackage.meanFilterA(img, 5, 5, True))
#######################################

#kernel=np.array([[1,2,3],[4,5,6],[7,8,9]])
#imRes=PIPpackage.imgConv(img,kernel)
#PIPpackage.printImgFloat(imRes)

#imgRes=PIPpackage.sepFilter([1,2,1,1,1],[1,2,1],img)

# res=PIPpackage.bilinear(1.6, 2.8, img,True)
#print('Bilinear interpolation result:', res)

# res=PIPpackage.medianHibrid5x5(4,5,img,True)

# img=PIPpackage.genImage(7,7,7,10)
# res=PIPpackage.eqHist(img,7)
# print('Equalized image')
# print(res)

#import matplotlib.pyplot as pyplot
#pyplot.imshow(img)
#pyplot.show()
#input("Press Enter to continue...")
#
# binImage=PIPpackage.genImage(6, 6, 1)
# print('Binary image')
# print(binImage)
# kernel = np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]])
# dilImage=PIPpackage.dilation(binImage,kernel)
# print('Dilated image')
# print(dilImage)
# erodImage=PIPpackage.erosion(binImage,kernel)
# print('Eroded image')
# print(erodImage)
#
# kernelHM = np.array([[-1, 0, -1], [-1, 0, -1], [-1, 0, 1]])
# hitMiss=PIPpackage.hitAndMiss(binImage,kernelHM)
# print('Hut and Miss image')
# print(hitMiss)

# img=np.array([[1,1,1,1,1,0,0,0,0,0,0,1,1,1,1],
#               [0,0,0,0,1,0,0,0,0,0,0,1,0,1,0],
#               [1,1,1,1,1,0,0,1,1,0,1,1,0,1,1],
#               [0,1,1,1,1,1,1,1,1,0,0,0,0,1,1],
#               [0,0,0,0,1,1,0,0,0,1,1,1,1,0,0],
#               [1,1,1,0,1,1,0,1,0,0,0,1,0,0,1],
#               [0,0,1,0,0,1,0,1,0,0,0,1,1,0,0],
#               [1,1,1,1,1,1,1,1,0,1,1,1,0,0,1]])
#
# labels,numLabels=PIPpackage.binSeg(img,True)
# print(numLabels)
# print(labels)

# img=np.array([[1,1,1,1,2,2,2,2,2,2,2,1,1,1,1],
#                [1,1,1,2,2,3,3,3,3,3,2,1,3,1,5],
#                [1,1,4,4,3,3,3,1,1,2,2,1,3,1,1],
#                [5,1,4,4,4,1,1,1,1,2,2,2,2,1,1],
#                [5,1,1,4,1,2,2,1,2,1,1,1,1,1,1],
#                [5,5,5,5,1,6,5,5,6,4,4,4,5,5,1],
#                [1,1,1,1,1,1,6,6,6,7,7,7,5,5,5],
#                [1,1,1,1,1,1,1,1,7,7,7,7,6,6,6]])
#
# labels,numLabels=PIPpackage.multiSeg(img,False)
# print(numLabels)
# print(labels)
