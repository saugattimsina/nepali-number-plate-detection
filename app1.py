from flask import Flask, render_template, request, redirect, url_for
from data import Articles
import os
from PIL import Image
from werkzeug.utils import secure_filename

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model
import os.path

import numpy as np
import cv2
from PIL import Image, ImageEnhance
import math
import shutil
from keras import backend as K

app = Flask(__name__)

Articles = Articles()
pic = []
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(APP_ROOT, 'static/images/')
def model(image):
    K.clear_session()
    model = load_model('numberplate3.h5')
    classes = model.predict_classes(image)
    K.clear_session()
    #classes = id2label(classes)
    return classes
'''def id2label(x):
    id2label = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "BA",11: "PA"}
    [x] = x
    classes = id2label["x"]
    return classes
'''


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    shutil.rmtree(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        newDes = os.path.join('static/images/'+filename)
        return roi_detection(newDes)
@app.route('/upload', methods=['POST'])
def roi_detection(image):
    pic.append(image)
    image = cv2.imread(image)
    image = cv2.resize(image,(1200,1200))
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=9)
    dilate = cv2.dilate(erosion, kernel, iterations=9)
    img = cv2.resize(dilate,(1200,1200))
    img = cv2.bilateralFilter(img, 9, 100, 100)
    median = cv2.medianBlur(img, 7)
    #cv2.imshow("med",median)

    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

    lower_red = np.array([140, 100, 110])
    upper_red = np.array([240, 255, 255])

    #img = cv2.resize(img,(1000,1000))
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(hsv, image, mask=mask)
    res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    cv2.imwrite("static/images/res.jpg",res)
    newRes = os.path.join('static/images/'+"res.jpg")
    pic.append(newRes)
    binary = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv2.imshow("image",image)
    #cv2.imshow("mask",res)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #cv2.imshow("r",binary)


    a=[]
    x_max= []
    y_max = []
    xv = []
    yv = []
    countrect=[]
    finalimg=[]
    for cnt in contours:
        if cv2.contourArea(cnt)>5000:
            x, y, w, h = cv2.boundingRect(cnt)

            a.append(cv2.contourArea(cnt))
            print(len(a))
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                countrect.append(cv2.contourArea(cnt))
    print(len(a))
    for contour in contours:
        if cv2.contourArea(contour) > 5000:
            x, y, w, h = cv2.boundingRect(contour)


            if len(a)>1 and len(countrect)==1:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                print(len(approx))
                if len(approx) == 4:
                    cv2.rectangle(image,(x,y),(x+w,y+h),(200,0,0),5)
                    final_image = image[y:y + h, x:x + w]
                    finalimg.append(final_image)

            elif len(a)>1 and (len(countrect) != 1):
                minvalueX=[]
                maxvalueX=[]
                minvalueY=[]
                maxvalueY=[]
                finalh=[]
                finalw=[]
                max_x = x + w
                max_y = y + h
                x_max.append(max_x)
                y_max.append(max_y)
                xv.append(x)
                yv.append(y)
                maxvalueX.append(max(max(x_max), max(xv)))
                minvalueX.append(min(min(x_max), min(xv)))
                minvalueY.append(min(min(y_max), min(yv)))
                maxvalueY.append(max(max(y_max), min(yv)))
                finalh.append(maxvalueY[0] - minvalueY[0])
                finalw.append(maxvalueX[0] - minvalueX[0])
                #cv2.rectangle(image, (minvalueX[0], minvalueY[0]), (minvalueX[0] + finalw[0], minvalueY[0] + finalh[0]), (0, 0, 255), 5)
                final_image = image[minvalueY[0]:minvalueY[0] + finalh[0], minvalueX[0]:minvalueX[0] + finalw[0]]
                finalimg.append(final_image)
                break


            elif(len(a)==1):
                #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
                final_image=image[y:y+h,x:x+w]
                finalimg.append(final_image)


            else:
                print("no contour found")


    #print(finalimg)
    x= finalimg[0]
    cv2.imwrite("static/images/final_image.jpg",x)
    final_imageRes = os.path.join('static/images/'+"final_image.jpg")
    upPlate = ""

    segmented = connect(final_imageRes)
    for i in range(len(segmented)):
        img = plt.imread("static/images/roi"+str(i)+".jpg")
        image = cv2.resize(img,(50,50))
        image = image.reshape(-1, image.shape[0],50, 1)
        label = model(image)
        [lab] = label
        print(lab)
        if lab==10:
            upPlate = upPlate + "बा "
            #print(upPlate)
        elif lab == 11:
            upPlate = upPlate + "प "
            #print(upPlate)
        else:
            if i ==2:
                upPlate = upPlate + str(lab) +" "
            else:
                upPlate = upPlate + str(lab)
            #print(upPlate)

        #print(leb)


    pic.append(final_imageRes)
    #cv2.imshow("rest",x)
    #cv2.imwrite("final.jpg",x)
    cv2.waitKey(0)
    return render_template('about.html',photos = pic, segLists = segmented,upPlate=upPlate)
#connect(os.path.join('static/images/'+"final_image.jpg"))
def connect(image):

    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print("hello")

    img = cv2.resize(img, (1000, 1000))

    #cv2.imshow("img",img)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=3)
    dilate = cv2.dilate(erosion, kernel, iterations=3)
    img = cv2.bilateralFilter(dilate, 9, 80, 80)

    median = cv2.medianBlur(img, 5)
    binary = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imwrite("static/images/binary.jpg",binary)
    bin = os.path.join('static/images/'+"binary.jpg")

    # getting mask with connectComponents
    nlabel,labels,stats,centroids = cv2.connectedComponentsWithStats(binary,connectivity=8)
    mask1=[]
    plate=[]
    plate1=[]
    tupple=()
    segmented = []
    for l in range(1,nlabel):
        if stats[l, cv2.CC_STAT_AREA] >= 2000:

            mask = np.array(labels, dtype=np.uint8)
            mask[labels == l] = 255
            mask1.append(mask)

            cv2.waitKey(0)

    for i in range(len(mask1)):
        #print(i)
        p = mask1[i]
        m = mask1[i]

        m = cv2.bilateralFilter(m, 19, 300, 300)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(m, kernel, iterations=4)
        m = cv2.dilate(erosion, kernel, iterations=2)
        m = cv2.medianBlur(m, 5)
        opening = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        thresh = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        #cv2.imshow("t",thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            cv2.drawContours(m, c, -1, (0, 0, 255), 3)
            x,y,w,h = cv2.boundingRect(c)

            if (h>150 and h<600) and (w>75 and w<600) :
                cv2.rectangle(m, (x, y), (x + w, y + h), (255, 255, 0), 2)

                plate.append([x,y,w,h])
                plate1.append(plate)
                tupple=tuple(plate1)
                plate1.clear()


            cv2.waitKey(0)
    listt=list(tupple)
    [final_list]=listt
    upper_plate=final_list[0:4]
    lower_plate=final_list[4:]

    #print(upper_plate)
    def sortSecond(val):
        return val[0]
    upper_plate.sort(key = sortSecond)
    lower_plate.sort(key=sortSecond)
    seg = []

    for i in range(len(upper_plate)):
        p=binary[upper_plate[i][1]:(upper_plate[i][1])+(upper_plate[i][3]),upper_plate[i][0]:(upper_plate[i][0])+(upper_plate[i][2])]
        segmented.append(p)
    print("segmented",segmented)
    for i in range(len(lower_plate)):
        p=binary[lower_plate[i][1]:(lower_plate[i][1])+(lower_plate[i][3]),lower_plate[i][0]:(lower_plate[i][0])+(lower_plate[i][2])]
        segmented.append(p)
    for i in range(len(segmented)):
        oo = segmented[i]
        print("i am oo",oo)
        cv2.imwrite('static/images/roi'+str(i)+'.jpg', oo)
        final_image_k = os.path.join('static/images/'+'roi'+str(i)+'.jpg')
        seg.append(final_image_k)
    print("before",len(seg))

    cv2.waitKey(0)
    print("i am seg",seg)
    return seg



if __name__ == '__main__':
    app.run(debug=True)
