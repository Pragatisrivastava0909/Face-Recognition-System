import cv2
from skimage import feature
import matplotlib.pyplot as plt
import os
import numpy as np
import csv
from sklearn import datasets,metrics,svm
import matplotlib.image as mimg
from sklearn.externals import joblib
import pandas as pd
from skimage import feature
import datetime

num_of_sample=8
name=input('Enter your name')

path1='%s'%(name)
if not os.path.exists(path1):
    os.makedirs(path1)
l=[]
l.append(name)
with open('myscsv.csv','w') as csvfile:
    fieldnames=[l[0],l[1],l[2]]
    writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
    writer.writeheader()
vid= cv2.VideoCapture(0)#to open the camera
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
#nose_cascade=cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
iter1=0
while(iter1<num_of_sample):
    r,frame=vid.read()# capture a single frame
    
    if r==True:
        frame= cv2.resize(frame,(640,480))
        im1=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        face=face_cascade.detectMultiScale(im1)
        #nose=nose_cascade.detectMultiScale(im1)
        eye=eye_cascade.detectMultiScale(im1)
        for x,y,w,h in (face):
            cv2.rectangle(frame,(x,y),(x+w,y+h),[255,0,0],5)
            iter1= iter1+1
            im_f= im1[y:y+h,x:x+w]
            im_f=cv2.resize(im_f,(112,92))
            cv2.putText(frame,'face no. '+str(iter1),(x,y),cv2.FONT_ITALIC,1,(255,0,255),2,cv2.LINE_AA)
            path2='./image/%d.png'%(iter1)
            cv2.imwrite(path2,im_f)
            path3='./%s/%d.png'%(name,iter1)
            cv2.imwrite(path3,im_f)
        cv2.imshow('frame',frame)
        cv2.waitKey(0)

vid.release()
cv2.destroyAllWindows()
q=len(l)
k=q*8
dataset=np.zeros((k,8748))
data_label=np.zeros((k,))

count=0
for i in range(1,q+1):
    for j in range(1,9):
        s=l[i-1]
        path='C:/Users/user/Desktop/pragati/python/%s/%d.png'%(s,j)   
        x=mimg.imread(path)
        y=feature.hog(x)

        f=y.reshape(1,-1)
        count=count+1
        dataset[count-1,:]=f
        data_label[count-1]=i
            

svm_model=svm.SVC(kernel='linear',gamma='auto',verbose=False)
svm_model= svm_model.fit(dataset,data_label)

joblib.dump(svm_model,'svm_model_face.pkl')


num_of_sample=50
vid= cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade=cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

iter1=0
a=joblib.load('svm_model_face.pkl')
    
while(iter1<num_of_sample):
    r,frame=vid.read()
    if r==True:
        frame= cv2.resize(frame,(640,480))
        im1=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        face=face_cascade.detectMultiScale(im1)
        nose=nose_cascade.detectMultiScale(im1)
        eye=eye_cascade.detectMultiScale(im1)
        for x,y,w,h in (face):
            cv2.rectangle(frame,(x,y),(x+w,y+h),[255,0,0],5)
            iter1= iter1+1
            im_f= im1[y:y+h,x:x+w]
            im_f=cv2.resize(im_f,(112,92))
            m=feature.hog(im_f)
            m=np.asarray(m)

            f=m.reshape(1,-1)
            b=a.predict(f) 
            n=l[int(b-1)]
            
            with open('./test data/%s.csv'%(i),'a') as f:
                thewriter=csv.writer(f)
                now= time.strftime('%d-%m-%y %H:%M:%S')
                thewriter.writerow([now])
            
            cv2.putText(frame,'face of '+n,(x,y),cv2.FONT_ITALIC,1,(255,0,255),2,cv2.LINE_AA)
            path2='./image/%d.png'%(iter1)
            cv2.imwrite(path2,im_f)
        cv2.imshow('frame',frame)
        cv2.waitKey(5)

vid.release()
cv2.destroyAllWindows()
