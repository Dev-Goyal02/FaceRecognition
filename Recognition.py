import cv2 as cv
import numpy as np
import os
import face_recognition
import pickle
import time
path='Actors3'
images=[]
names=[]
myList=os.listdir(path)
print(myList)
for cls in myList:
    curImg=cv.imread(f'{path}/{cls}')
    images.append(curImg)
    names.append(os.path.splitext(cls)[0])
print(names)
with open('Encodings_separate.dat', 'rb') as f:
	Trained_Encodings = pickle.load(f)

encodeListKnown=Trained_Encodings

cap=cv.VideoCapture(0)
while True:
    success,img=cap.read()
    #imgL=cv.resize(img,(int(2*img.shape[1]),int(2*img.shape[0])))
    imgS=cv.resize(img,(0,0),None,0.25,0.25)
    #imgS=img
    imgS=cv.cvtColor(imgS,cv.COLOR_BGR2RGB)
    facesCurrFrame=face_recognition.face_locations(imgS)
    encodeCurrFrame=face_recognition.face_encodings(imgS,facesCurrFrame)
    
    #Comparing faces obtained through webcam frame by our known list of images
    for encodeFace,FaceLoc in zip(encodeCurrFrame,facesCurrFrame):
        match=face_recognition.compare_faces(encodeListKnown,encodeFace,tolerance=0.5)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        confidence=min(faceDis)
        percentage=100-round(confidence,2)*100
        detected=np.argmin(faceDis)
        #Printing the name if we found a true value of face being matched
        if match[detected]:
            name_matched=names[detected]
            #print(name_matched)
            y1,x2,y2,x1=FaceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),-1)
            cv.putText(img,name_matched,(x1+6,y2-6),cv.FONT_HERSHEY_COMPLEX
            ,1,(255,0,0),2)
            print(f'{name_matched} with confidence of {percentage}%')
            with open('Records.txt','a') as f:
                f.writelines(f'\n{name_matched} : {time.asctime(time.localtime(time.time()))}')
    cv.imshow('WEBCAM',img)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cap.destroyAllWindows()