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

app = Flask(__name__)

Articles = Articles()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/articles')
def articles():
    return render_template('articles.html', articles=Articles)


@app.route('/article/<string:id>')
def article(id):
    return render_template('article.html', articles=Articles, id=id)


@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'static/images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)

    photos = []
    newDes = os.path.join('static/images/'+filename)
    photos.append(newDes)
    image = cv2.imread(newDes)
    img = cv2.bilateralFilter(image, 9, 75, 75)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([140, 100, 110])
    upper_red = np.array([240, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    # maskLoc = os.path.join('images/'+mask)
    res = cv2.bitwise_and(hsv, image, mask=mask)
    cv2.imwrite("static/images/res.jpg",res)
    newRes = os.path.join('static/images/'+"res.jpg")
    photos.append(newRes)
    # cv2.imshow("res",res)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # cv2.imshow("hellll",mask)

    x_max= []
    y_max = []
    xv = []
    yv = []
    pts = []
    for c in contours:
        if cv2.contourArea(c) > 1200:
            x, y, w, h = cv2.boundingRect(c)
            max_x = x+w
            max_y = y+h
           # if max_x < (max_y+4/3*max_y):
            #    max_x = math.floor((max_y+4/3*max_y))
            #else:
             #   print("hello")
            x_max.append(max_x)
            y_max.append(max_y)
            xv.append(x)
            yv.append(y)

    maxvalueX = max(max(x_max),max(xv))
    minvalueX = min(min(x_max),min(xv))
    minvalueY = min(min(y_max),min(yv))
    maxvalueY = max(max(y_max),min(yv))
    finalh = (maxvalueY-minvalueY)
    finalw = (maxvalueX - minvalueX)

    cv2.rectangle(image,(minvalueX,minvalueY),(minvalueX+finalw,minvalueY+finalh),(29,0,255),1)

    final_image = image[minvalueY:minvalueY+finalh,minvalueX:minvalueX+finalw]

    cv2.imwrite("static/images/final_image.jpg",final_image)
    final_imageRes = os.path.join('static/images/'+"final_image.jpg")
    photos.append(final_imageRes)
    print(photos)

    # cv2.imwrite("checkcheck.jpg", final_image)
    image = cv2.imread(final_imageRes, cv2.IMREAD_UNCHANGED);

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel1 = np.ones((2,2),np.uint8)
    erosion = cv2.erode(gray, kernel1, iterations = 2)
    binary = cv2.threshold(erosion, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # getting mask with connectComponents
    ret, labels = cv2.connectedComponents(binary)
    #cv2.imshow("r",ret)
    for label in range(1,ret):
        mask = np.array(labels, dtype=np.uint8)
        mask[labels == label] = 255

        #cv2.imshow('component',mask)
    listx = []
    listy = []
    listw = []
    listh = []
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # getting ROIs with findContours
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        p = cv2.contourArea(cnt)
        if p>500:
            if h>20 and w>20 and h<100 and w<100:
                #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,0))
                listx.append(x)
                listy.append(y)
                listw.append(w)
                listh.append(h)
    segLists =[]
    for i in range(len(listx)):
        final = image[listy[i]:listy[i] + listh[i], listx[i]:listx[i] + listw[i]]
        cv2.imwrite('static/images/roi'+str(i)+'.jpg', final)
        final_image_k = os.path.join('static/images/'+'roi'+str(i)+'.jpg')
        segLists.append(final_image_k)

    print(segLists)

    downPlate = ''
    upPlate = ''
    for i in range(len(segLists)):
        img = plt.imread("static/images/roi"+str(i)+".jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3,3), np.uint8)
        image = cv2.erode(gray, kernel, iterations=1)
        #   image = cv2.dilate(gray, kernel, iterations=1)

        image = cv2.resize(image,(50,50))

        image = image.reshape(-1, image.shape[0],50, 1)

        model = load_model('numberplate1.h5')
        classes = model.predict_classes(image)

        if classes<10:
            print(i)
            if i>=0 and i<=3:
                downPlate = downPlate+ str(classes)
            else:
                upPlate = upPlate+ str(classes)

        elif classes==10:
            upPlate = upPlate+ "BA"

        else:
            upPlate = upPlate+ "PA"

    print("dp",downPlate)
    print("up",upPlate)
    # cv2.imshow("mask.png", mask)

    # cv2.imshow("initial", image)

    return render_template('about.html',photos = photos, segLists = segLists, upPlate=upPlate, downPlate= downPlate)
    #return (results, destination)




if __name__ == '__main__':
    app.run(debug=True)
