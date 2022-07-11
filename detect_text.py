import cv2
import os
import re
import pytesseract
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#Hàm duyệt ảnh bằng tesseract
def ReadModel(image,language="vie"):
	plt.figure()
	plt.imshow(image,cmap='gray', vmin=0, vmax=255)
	text = pytesseract.image_to_string(image,language)
	#Improve output
	text = re.sub(r'(\s,\.,\:){2,}', ' ', text)
	text = text.lstrip('.').lstrip(',').lstrip('_').rstrip('.').rstrip(',').lstrip("“")
	text = re.sub(r'\_\={1}', '', text)
	return text
    
#Các hàm tiền xử lý trước khi đưa vào model	
def detect(img):
	dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,10,25)
	gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(gray)
	thresh = cv2.adaptiveThreshold(cl1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,30)
	return ReadModel(thresh)

def detect_number(img):
	dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,10,25)
	gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
	closedDigit= cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, 1)
	thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,32)
	return ReadModel(thresh,"eng")

def detect_id_number(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    im = cv2.filter2D(img, -1, kernel)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return ReadModel(gray,"eng")

def detect_id(img):
    dst = cv2.fastNlMeansDenoisingColored(img,None,5,5,4,9)
    kernel1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    im = cv2.filter2D(dst, -1, kernel1)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return ReadModel(gray,"vie")

def detect_back(img):
	dst = cv2.fastNlMeansDenoisingColored(img,None,5,5,4,15)
	gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
	closedDigit= cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, 1)
	thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,30)
	return ReadModel(thresh)

import os
import cv2
import crop_img as crop
import get_detail as get
import matplotlib.pyplot as plt
#Preprocessing
def Processing(typed,info):
    if typed == 1:
        number= detect_number(info[0])
        name=detect(info[1]) +" "+detect(info[2])
        birth_date=detect_number(info[3])
        birth_place= detect(info[4])+" "+detect(info[5])
        lived= detect(info[6])+" "+detect(info[7])
        print(u"Số CMND: ", number)
        print(u"Tên: ", name)
        print(u"Ngày sinh: ", birth_date)
        print(u"Nơi sinh: ", birth_place)
        print(u"Hộ khẩu: ", lived)
        plt.figure()
        plt.imshow(info[8])
        number, name, birth_date, gender, quoc_tich, que_quan1,que_quan2, ho_khau1, ho_khau2,valid
    elif typed == 2:
        number= detect_id_number(info[0])
        name=detect_id(info[1])
        birth_date=detect_id_number(info[2])
        gender= detect_id(info[3])
        quoc_tich= detect_id(info[4])
        que_quan= detect_id(info[5])+" "+detect_id(info[6])
        ho_khau= detect_id(info[7])+" "+detect_id(info[8])
        # valid= detect(info[9])
        print(u"Số Căn cước công dân: ",number)
        print(u"họ và Tên: ",name)
        print(u"Ngày, tháng, năm sinh: ",birth_date)
        print(u"giới tính: ",gender)
        print(u"Quốc tịch: ",quoc_tich)
        print(u"Quê quán : ",que_quan)
        print(u"Nơi thường trú: ",ho_khau)
    elif typed in (3,4):
        print("Vẫn chưa có thông tin để nhận diện")
    
    # elif typed == 4 or :