import PIPpackage
import numpy as np
#
# #########
# # Exame generation functions
# #################
#
# def getTxt(vals,images):
#     f = open(os.path.join(filesPath,"template 1 teste.txt"), "r")
#     s = f.read()
#     for i in range(len(vals)):
#         s=s.replace(f'<{i+1}>',str(vals[i]))
#     for i in range(len(images)):
#         s=s.replace(f'<image{i+1}>',images[i])
#     return(s)
#
# def randof(list):
#     i=random.randint(0,len(list)-1)
#     return([list[i]])
#
# def genVals(studentNumb):
#     random.seed(studentNumb)
#     vals=[]
#     vals += randof([2,3])
#     vals += randof([2,3])
#     vals += [random.randint(5,20)]
#     vals += randof([1, 2, 3])
#     vals += randof([1, 2, 3])
#     vals += randof([0, 4])
#     vals += randof([1, 2, 3])
#     vals += [random.randint(520, 620)]
#     return(vals)
#
# ##########
# # Generate exam
# ####################
#
# def genExam(studentNumb,name):
#
#     try:
#         os.mkdir(os.path.join(examsPath,str(studentNumb)))
#     except:
#         pass
#     try:
#         os.mkdir(os.path.join(solutionsPath,str(studentNumb)))
#     except:
#         pass
#
#     os.chdir(os.path.join(examsPath,str(studentNumb)))
#
#     original_stdout=sys.stdout
#     with open(str(studentNumb)+'_exame_SS.txt', 'w') as f:
#         sys.stdout = f
#         img1,imlist1=genImage(5,127)
#         img2,imlist2=genImage(6,7)
#         while img2==eqHist(imlist2,8,False):
#             img2, imlist2 = genImage(6, 7)
#         img3, imList3 = genImage(6, 1)
#         while not numBits(imList3) in [8, 12]:
#             img3, imList3 = genImage(6, 1)
#         kerHitMiss = [[-1, 0, -1], [-1, 0, -1], [-1, 0, 1]]
#         img4, imList4 = genImage(6, 1)
#         while not numBits(hitAndMiss(imList4, kerHitMiss)) in [8, 12]:
#             img4, imList4 = genImage(6, 1)
#
#         vals=genVals(studentNumb)
#         txt=getTxt(vals,[img1,img2,img3,img4])
#         txt=txt.replace('<numero>',str(studentNumb))
#         txt=txt.replace('<nome>',name)
#         print(txt)
#         f.close()
#         copyfile(os.path.join(examsPath,str(studentNumb),str(studentNumb)+'_exame_SS.txt'),
#                                 os.path.join(solutionsPath,str(studentNumb),str(studentNumb)+'_exame_SS.txt'))
#
#     sys.stdout = original_stdout
#
# #### Corretion generation #####
#
#     os.chdir(os.path.join(solutionsPath,str(studentNumb)))
#     with open(str(studentNumb)+'_exame_SS_sol.txt', 'w') as f:
#         sys.stdout = f
#         print('\nAluno:',name,'  Numero:',studentNumb)
#     #2 Kernal separado
#         print('PERGUNTA 2')
#         imRes = sepFilter3x3([2,3,2], [2,6,2], imlist1)
#         print('Valor do pixel (2,2):','{:6.3f}'.format(imRes[2][2]))
#         print('Valor do pixel (4,3):','{:6.3f}'.format(imRes[3][4]))
#     #3 Rotacao
#         print('PERGUNTA 3')
#         printImg(imlist1)
#         xorig, yorig = calcRotCoord(vals[0], vals[1], vals[2])
#         res = bilinear(xorig, yorig, imlist1)
#         print('Final result', res)
#     #4 Filtro Mediana Hibrido 5x5
#         print('PERGUNTA 4')
#         medianHibrid5x5(vals[3],vals[4],imlist1,True)
#         medianHibrid5x5(vals[5],vals[6],imlist1,False)
#     #5 Equializacao de histograma
#         print('PERGUNTA 5')
#         printImg(imlist2)
#         eqHist(imlist2,8,True)
#     #6 Fecho numa imagem binaria com kernel
#         print('PERGUNTA 6')
#         ker = [[0, 1, 0], [0, 1, 1], [1, 0, 1]]
#         print('Kernel')
#         printImg(ker)
#         print('Original image')
#         printImg(imList3)
#         imResD = dilation(imList3, ker)
#         print('Dilated image')
#         printImg(imResD)
#         imResE = erosion(imResD, ker)
#         print('Eroded image')
#         printImg(imResE)
#     # Fecho numa imagem binaria com kernel
# #        print('PERGUNTA 6')
# #        print('Input binary image')
# #        printImg(imList4)
# #        print('Hit & Miss kernel')
# #        printImg(kerHitMiss)
# #        imResHM = hitAndMiss(imList4, kerHitMiss)
# #        print('Hit & Miss resulting image')
# #        printImg(imResHM)
#     # Sensores
#         print('PERGUNTA 7')
#         readings = [[20.6, 26], [31.1, 192], [32.5, 213], [33.3, 218], [34.2, 274], [39.6, 294], [40.0, 380],
#                     [48.3, 413], [48.8, 454], [49.2, 479], [49.0, 500], [58.6, 640], [59.4, 641], [60.5, 668],
#                     [63.2, 679], [63.6, 681], [64.1, 732]]
#
#         sensorV = vals[7]
#         realV = 55
#
#         upper = 0
#         while readings[upper][1] < sensorV:
#             upper += 1
#         for j in range(upper + 1, len(readings)):
#             dif = readings[j][1] - sensorV
#             if dif > 0 and dif < readings[upper][1] - sensorV:
#                 upper = j
#
#         lower = 0
#         while readings[lower][1] > sensorV:
#             lower += 1
#         for j in range(lower + 1, len(readings)):
#             dif = sensorV - readings[j][1]
#             if dif > 0 and dif < sensorV - readings[lower][1]:
#                 lower = j
#
#         print('Sensor value:', sensorV)
#         print('Lower value:', readings[lower][1], readings[lower][0])
#         print('Higher value:', readings[upper][1], readings[upper][0])
#         off = (sensorV - readings[lower][1]) / (readings[upper][1] - readings[lower][1])
#         print('Offset:', '{:5.3f}'.format(off))
#         v = interp(readings[lower][0], readings[upper][0], off)
#         print('Absolute error:','{:5.2f}'.format(v - realV))
#         print('Relative error:'+'{:5.2f}'.format(abs(v - realV) / realV * 100) + '%')
#         f.close()
#
#     sys.stdout = original_stdout
#
# def makeFolders():
#     os.chdir(filesPath)
#     try:
#         os.mkdir('Exams')
#     except:
#         pass
#     try:
#         os.mkdir('Solutions')
#     except:
#         pass
#
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

#res=PIPpackage.eqHist(img,7,True)
#print('Equalized image')
#print(res)

#import matplotlib.pyplot as pyplot
#pyplot.imshow(img)
#pyplot.show()
#input("Press Enter to continue...")

binImage=PIPpackage.genImage(6, 6, 1)
print('Binary image')
print(binImage)
kernel = np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]])
dilImage=PIPpackage.dilation(binImage,kernel)
print('Dilated image')
print(dilImage)
erodImage=PIPpackage.erosion(binImage,kernel)
print('Eroded image')
print(erodImage)