import cv2 as cv
import numpy as np
haar_cascade = cv.CascadeClassifier('haar_face.xml')
name = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
features =np.load('feature.npy',allow_pickle=True)
lables = np.load('lable.npy') 
img = cv.imread("res\\Faces\\val\\mindy_kaling\\2.jpg")
#cv.imshow('Detected',img )

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainedFaces.yml')
gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
faces = haar_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces :
    face_roi = gray[y:y+h,x:x+y]
    lable,confidance =face_recognizer.predict(face_roi)
    print(f'the Confidence level is : {confidance}')
    cv.putText(img ,str(name[lable]),(x,y),cv.FONT_HERSHEY_COMPLEX_SMALL,1.0,(200,199,0),thickness=1)
    cv.rectangle(img ,(x,y),(x+w,y+h),(0,220,0),thickness=2)
cv.imshow('Detected',img )
cv.waitKey(0)