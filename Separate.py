import cv2 as cv
import numpy as np
import os
import face_recognition
import pickle
path='Actors3'
images=[]
names=[]
#Creating a myList variable with Path of all the images
myList=os.listdir(path)
print(myList)
#Creating a list names of all the images

for cls in myList:
    curImg=cv.imread(f'{path}/{cls}')
    images.append(curImg)
    names.append(os.path.splitext(cls)[0])
print(names)

#Finding encodings of all the images
encodeListKnown=[]
def find_encodings(images,encodeList):
    for img in images:
        #gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        #gray=cv.cvtColor(gray,cv.COLOR_BGR2RGB)
        #faceLoc=face_recognition.face_locations(gray)[0]
        #encodeLoc=face_recognition.face_encodings(gray)[0]
        img=cv.resize(img,(0,0),None,0.75,0.75)
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        #img=face_recognition.load_image_file(curr)
        NumberOfFaces=face_recognition.face_locations(img)
        if NumberOfFaces:
            faceLoc=NumberOfFaces[0]
            encodeLoc=face_recognition.face_encodings(img)[0]
        #cv.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),
        #(0,255,0),2)
            encodeList.append(encodeLoc)

find_encodings(images,encodeListKnown)
print('Encoding Complete')
with open('Encodings_separate.dat', 'wb') as f:
    pickle.dump(encodeListKnown, f)

# path='Numbered Images'
# TestList=os.listdir(path)
# for test in TestList:
#     cur=cv.imread(f'{path}/{test}')
#     resized=cv.resize(cur,(0,0),None,0.5,0.5)
#     #resized_gray=cv.cvtColor(resized,cv.COLOR_BGR2GRAY)
#     #resized_gray=cv.cvtColor(resized_gray,cv.COLOR_BGR2RGB)
#     #facesCurrFrame=face_recognition.face_locations(resized_gray)
#     #encodeCurrFrame=face_recognition.face_encodings(resized_gray,facesCurrFrame)
#     resized=cv.cvtColor(resized,cv.COLOR_BGR2RGB)
#     #resized=face_recognition.load_image_file(resized_cur)
#     facesCurrFrame=face_recognition.face_locations(resized)
#     encodeCurrFrame=face_recognition.face_encodings(resized,facesCurrFrame)
#     for encodeFace,FaceLoc in zip(encodeCurrFrame,facesCurrFrame):
#         matches=face_recognition.compare_faces(encodeListKnown,encodeFace,tolerance=0.5)
#         faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
#         #print(faceDis)
#         confidence=min(faceDis)
#         percentage=round(confidence,2)*100
#         matchIndex=np.argmin(faceDis)
#         if matches[matchIndex]:
#             name_matched=names[matchIndex]
#             id=os.path.splitext(test)[0]
#             print(f'{id}->{name_matched} with confidence of {100-percentage}%')
            