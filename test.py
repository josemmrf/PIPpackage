import PIPpackage
import numpy as np

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

#img=PIPpackage.genImage(7,7,7,10)
#PIPpackage.printImg(img)
#kernel=np.array([[1,2,3],[4,5,6],[7,8,9]])
#imRes=PIPpackage.imgConv(img,kernel)
#PIPpackage.printImgFloat(imRes)

#imgRes=PIPpackage.sepFilter([1,2,1,1,1],[1,2,1],img)

# res=PIPpackage.bilinear(1.6, 2.8, img,True)
#print('Bilinear interpolation result:', res)

# res=PIPpackage.medianHibrid5x5(4,5,img,True)

img=PIPpackage.genImage(7,7,7,10)
res=PIPpackage.eqHist(img,7,True)
print('Equalized image')
print(res)

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

img=np.array([[1,1,1,1,1,0,0,0,0,0,0,1,1,1,1],
              [0,0,0,0,1,0,0,0,0,0,0,1,0,1,0],
              [1,1,1,1,1,0,0,1,1,0,1,1,0,1,1],
              [0,1,1,1,1,1,1,1,1,0,0,0,0,1,1],
              [0,0,0,0,1,1,0,0,0,1,1,1,1,0,0],
              [1,1,1,0,1,1,0,1,0,0,0,1,0,0,1],
              [0,0,1,0,0,1,0,1,0,0,0,1,1,0,0],
              [1,1,1,1,1,1,1,1,0,1,1,1,0,0,1]])

labels,numLabels=PIPpackage.binSeg(img,True)
print(numLabels)
print(labels)

#img=np.array([[1,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
#              [1,0,0,0,0,0,0,0,0,0,0,1,0,1,0],
#              [1,1,0,0,0,0,0,1,1,0,0,1,0,1,1],
#              [0,1,1,0,0,1,1,1,1,0,0,0,0,1,1],
#              [0,0,1,0,1,0,0,1,0,1,1,1,1,1,1],
#              [1,1,1,0,1,0,0,1,0,0,0,0,0,0,1],
#              [1,0,0,0,1,1,0,1,0,0,0,0,0,0,0],
#              [1,0,0,0,0,1,1,1,0,0,0,0,0,0,0]])

#labels,numLabels=PIPpackage.binSeg(img,True)
#print(numLabels)
#print(labels)
