import os
import cv2 as cv
import numpy as np

DIR = r'D:\Workstation\OpenCV\res\Faces\train' #address where faces a save
haar_cascade = cv.CascadeClassifier('haar_face.xml')
#getting the name of faces by folders
peoples= []
for i in os.listdir(r'D:\Workstation\OpenCV\res\Faces\train'):
    peoples.append(i)
print(peoples)
#traning function 
feature = []
lables = []
def create_train():
    #itrate to every folder
    for person in peoples :
        path = os.path.join(DIR,person)
        lable = peoples.index(person)
        for img in os.listdir(path): #getting the list of members in path directory
            img_path = os.path.join(path,img) #getting every single image
            img_arry =cv.imread(img_path) #reading image
            gray = cv.cvtColor(img_arry,cv.COLOR_RGB2GRAY)
            faces = haar_cascade.detectMultiScale(gray, 1.1, 5)
            for (x,y,w,h) in faces :
                faces_roi = gray[x:x+w,y:y+h] #regoin of intrest which have to crop and same
                feature.append(faces_roi)
                lables.append(lable)
create_train()

print("Baby ______________ it trained")

#print(f'No of feature in feature list is {len(feature)}')
#print(f'No of Lable in feature list is {len(lables)}')

faceRegonizer = cv.face.LBPHFaceRecognizer_create()
#coverting feature & label list into numpy array 
lables =np.array(lables)
feature = np.array(feature , dtype='object')
 
#train recognizeron feature and lable list
faceRegonizer.train(feature,lables)
#saving the yml and npy file for dont repeat whole process
np.save("feature.npy",feature)
np.save("lable.npy",lables)
faceRegonizer.save("trainedFaces.yml")
        

            



