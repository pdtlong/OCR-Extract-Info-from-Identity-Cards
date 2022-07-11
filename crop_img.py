import os
import cv2
import math
import imutils
import numpy as np
from PIL import Image
# import imutils
def set_image_dpi(file_path):
  im = Image.open(file_path)
  length_x, width_y = im.size
  factor = min(1, float(1024.0 / length_x))
  size = int(factor * length_x), int(factor * width_y)
  im_resized = im.resize(size, Image.ANTIALIAS)
  filename = "{}.png".format(os.getpid())
  im_resized.save(filename, dpi=(300, 300))
  return filename

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
  dim = None
  (h, w) = image.shape[:2]
  # Trả về ảnh gốc nếu ko cần tăng
  if width is None and height is None:
    return image
    # Nếu chiều rộng none
  if width is None:
    # Tính theo chiều cao
    r = height / float(h)
    dim = (int(w * r), height)
  else:
    # tim của with theo chiều
    r = width / float(w)
    dim = (width, int(h * r))

  resized = cv2.resize(image, dim, interpolation = inter)
  return resized

def order_points(pts):

  rect = np.zeros((4, 2), dtype = "float32")

  s = pts.sum(axis = 1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]

  diff = np.diff(pts, axis = 1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]
  return rect

def four_point_transform(image, pts):
  rect = order_points(pts)
  (tl, tr, br, bl) = rect

  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))

  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))

  # order
  dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")

  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

  # return the warped image
  return warped

def auto_canny(image, sigma=0.33):
  # compute the median of the single channel pixel intensities
  v = np.median(image)
  # apply automatic Canny edge detection using the computed median
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(image, lower, upper)
  return edged

def Back_Filter(img):
  card = img.copy()
  # resize using ratio (old height to the new height)
  ratio = img.shape[0] / 408
  card = imutils.resize(card, height=408)
  gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
  cl1 = clahe.apply(gray)
  edged = cv2.Canny(cl1, 180, 255,apertureSize = 3)
  #Closing
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
  closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
  #Find biggest retangle
  cmnd_ct = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cmnd_ct = imutils.grab_contours(cmnd_ct)
  #Duyệt qua tất cả các hình có 4 góc
  cmnd_ct = sorted(cmnd_ct, key = cv2.contourArea, reverse = True)[:5]
  # loop over the contours

  for c in cmnd_ct:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
      Cnt = approx
      diagonal1 = math.sqrt( ((Cnt[1][0][0]-Cnt[3][0][0])**2)+((Cnt[1][0][1]-Cnt[3][0][1])**2) )
      diagonal2 = math.sqrt( ((Cnt[0][0][0]-Cnt[2][0][0])**2)+((Cnt[0][0][1]-Cnt[2][0][1])**2) )
      x,y,width,height = cv2.boundingRect(Cnt)
      if diagonal1 >width and diagonal2 >width:
        warped = four_point_transform(img, Cnt.reshape(4, 2) * ratio)
        return warped
  return []

def Front_Filter(img):
  card = img.copy()
  # resize using ratio (old height to the new height)
  ratio = img.shape[0] / 408
  card = imutils.resize(card, height=408)
  blur = cv2.GaussianBlur(card, (3, 3), 0)
  hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
  green = cv2.inRange(hsv, (60, 30, 30), (100, 150,200))
  # Tìm 4 góc của CMND
  cmnd_ct = cv2.findContours(green.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cmnd_ct = imutils.grab_contours(cmnd_ct)
  #Duyệt qua tất cả các hình có 4 góc
  cmnd_ct = sorted(cmnd_ct, key = cv2.contourArea, reverse = True)[:5]
  # loop over the contours
  for c in cmnd_ct:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
      Cnt = approx
      x,y,width,height = cv2.boundingRect(Cnt)
      diagonal1 = math.sqrt( ((Cnt[1][0][0]-Cnt[3][0][0])**2)+((Cnt[1][0][1]-Cnt[3][0][1])**2) )
      diagonal2 = math.sqrt( ((Cnt[0][0][0]-Cnt[2][0][0])**2)+((Cnt[0][0][1]-Cnt[2][0][1])**2) )
      if diagonal1 >width and diagonal2 >width and width>=400:
        warped = four_point_transform(img, Cnt.reshape(4, 2) * ratio)
        return warped
  return []

def ID_Filter(img):
  card = img.copy()
  # resize using ratio (old height to the new height)
  ratio = img.shape[0] / 408
  card = imutils.resize(card, height=408)
  blur = cv2.GaussianBlur(card, (3, 3), 0)
  hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
  bw = cv2.inRange(hsv, (60, 0, 100), (130, 100,255))
  borderLen = 3 
  lenx, leny = bw.shape
  bw[0:borderLen,0:leny] = 0
  bw[lenx-borderLen:lenx,0:leny] = 0
  bw[0:lenx,0:borderLen] = 0
  bw[0:lenx,leny-borderLen:leny] = 0
  # Tìm 4 góc của CMND
  cmnd_ct = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cmnd_ct = imutils.grab_contours(cmnd_ct)
  #Duyệt qua tất cả các hình có 4 góc
  cmnd_ct = sorted(cmnd_ct, key = cv2.contourArea, reverse = True)[:5]
  for c in cmnd_ct:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) >= 4:
      Cnt = approx
      x,y,width,height = cv2.boundingRect(Cnt)
      diagonal1 = math.sqrt( ((Cnt[1][0][0]-Cnt[3][0][0])**2)+((Cnt[1][0][1]-Cnt[3][0][1])**2) )
      diagonal2 = math.sqrt( ((Cnt[0][0][0]-Cnt[2][0][0])**2)+((Cnt[0][0][1]-Cnt[2][0][1])**2) )
      if diagonal1 >width and diagonal2 >width and width>=400:
        warped = four_point_transform(img, Cnt.reshape(4, 2) * ratio)
        return warped
  return []

def Extend_Filter(img):
  card = img.copy()
  # resize using ratio (old height to the new height)
  ratio = img.shape[0] / 408.0
  card = imutils.resize(card, height=408)
  hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
  blur = cv2.GaussianBlur(hsv[:,:,1],(7 ,7),0)
  blured = cv2.bilateralFilter( blur, 5, 50, 50 )
  edges = cv2.Canny(blured,40,200)
  kernel = np.ones((5,5), np.uint8) 
  dilation = cv2.dilate(edges, kernel, iterations=1) 
  # Tìm 4 góc của CMND
  cmnd_ct = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cmnd_ct = imutils.grab_contours(cmnd_ct)
  #Duyệt qua tất cả các hình có 4 góc
  cmnd_ct = sorted(cmnd_ct, key = cv2.contourArea, reverse = True)[:5]
  # loop over the contours

  for c in cmnd_ct:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
      Cnt = approx
      x,y,width,height = cv2.boundingRect(Cnt)
      diagonal1 = math.sqrt( ((Cnt[1][0][0]-Cnt[3][0][0])**2)+((Cnt[1][0][1]-Cnt[3][0][1])**2) )
      diagonal2 = math.sqrt( ((Cnt[0][0][0]-Cnt[2][0][0])**2)+((Cnt[0][0][1]-Cnt[2][0][1])**2) )
      if diagonal1 >width and diagonal2 >width and width>=400:
        warped = four_point_transform(img, Cnt.reshape(4, 2) * ratio)
        return warped
  return []