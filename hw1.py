from PIL import Image
import cv2
import copy
import math
from scipy.signal import sepfir2d
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


sobelkernx_base = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
]

sobelkerny_base = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
]


'''returns a 2d gaussian kernel'''
def gkern(len=5, nsig=1):
    lim = len//2 + (len % 2)/2
    x = np.linspace(-lim, lim, len+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

def fix_from_png(img):
    img_fixed = [] 
    for i in range(0, img.shape[0]):
        img_row = []
        for j in range(0, img.shape[1]):
            img_row.append(img[i][j][0])
        img_fixed.append(img_row)
    return img_fixed

def isDifferent(img1, img2):
    return not img1 == img2

def divFilt(filt, div):
    new_filt = [len(filt)]
    for row in filt:
        new_row = []
        for item in row:
            new_row.append(item/div)
        new_filt.append(new_row)
    return new_filt

def filter(img, filt):
    height = len(img)
    width = len(img[0])
    filtered_img = np.zeros((height, width))
    half = math.floor(len(filt)/2)
    for i in range(0, height-1):
        width = len(img[i])
        for j in range(0, width-1):
            for x in range(-half, half):
                for y in range(-half, half):
                    if(i+x > 0 and i+x < height and j+y > 0 and j+y < width):
                        try:
                            new_val = filtered_img[i][j]+filt[x+half][y+half]*img[i+x][j+y]
                            if new_val < 0:
                                new_val = 0
                            if new_val > 255:
                                new_val = 255
                            filtered_img[i][j] = new_val
                        except IndexError:
                            print("ERROR: i = ", i, " j = ", j, " x = ", x, " y = ", y)
    return filtered_img

def toImage(img):
    img_arr = np.asarray(img)
    return Image.fromarray(img_arr.astype('uint8'))


img = cv2.imread('road.png')

img = fix_from_png(img)

#print(img)

kernel = gkern()
print(kernel)

img_gauss = filter(img, kernel)

#imgnp = np.array(img)
#imgnp_gauss = np.array(img_gauss)
#print(img_gauss)

#cv2.imshow('pre-gauss', img)
#cv2.imshow('gaussian', img_gauss)

#print(img == img_gauss)

img_arr = np.asarray(img)
img_gauss_arr = np.asarray(img_gauss)

ime = Image.fromarray(img_arr)
ime_guass = Image.fromarray(img_gauss_arr)

ime.show()
ime_guass.show()

sobelkernx = divFilt(sobelkernx_base, 9.0)
sobelkerny =divFilt(sobelkerny_base, 9.0)


sobelx = filter(img_gauss, sobelkernx)
sobely = filter(img_gauss, sobelkerny)

#print(sobelx)
toImage(sobelx).show()

#toImage(sobelx).show()
toImage(sobely).show()

#ime_guass.show()

#cv2.waitKey(0)

#cv2.destroyAllWindows()
