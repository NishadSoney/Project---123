import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.datasets import fetch_openml 
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
import os,ssl,time

if(not os.environ.get("PYTHONHTTPSVERIFY","")and
    getattr(ssl,"_create_unverified_context",None)):
    ssl._create_default_https_context = ssl._create_unverified_context
x,y = fetch_openml("mnist_784",version = 1,return_X_y = True)
print(pd.Series(y).value_counts())
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nc = len(classes)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 9,train_size = 7500,test_size = 2500)
x_train_scale = x_train/255
x_test_scale = x_test/255

clf = LogisticRegression(solver = "saga",multi_class = "multinomial").fit(x_train_scale,y_train)

y_pre = clf.predict(x_test_scale)
acc = accuracy_score(y_test,y_pre)

print(acc)

cap = cv2.VideoCapture(1)
while(True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape()
        ul = (int(width/2-56),int(height/2-56))
        br = (int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,ul,br,(0,255,0),2)
        roi = gray[ul[1]:br[1],ul[0]:br[0]]
        iml_pil = Image.fromarray(roi)
        img = iml_pil.convert("L")
        img_r = img.resize((28,28),Image.ANTIALIAS)
        img_r_i = PIL.ImageOps.invert(img_r)
        p_f = 20
        minp = np.percentile(img_r_i,p_f)
        img_r_i_scale = np.clip(img_r_i-minp,0,0,255)
        max_p = np.max(img_r_i)
        img_r_i_scale = np.asanyarray(img_r_i_scale)/max_p
        test_s = np.array(img_r_i_scale).reshape(1,784)
        test_pre = clf.predict(test_s)
        print(test_pre)
        cv2.imshow("frame",gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.destoryAllWindows()