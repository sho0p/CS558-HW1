from PIL import Image
import cv2
from scipy.signal import sepfir2d
import matplotlib.pyplot as plt
import numpy as np


'''returns a 2d gaussian kernel'''
def gkern(len=5, nsig=1):
    ax = np.arange(-len // 2+1.//2+1.)
    xx,yy = np.meshgrid(ax,ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(nsig))

    return kernel/np.sum(kernel)

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

def filter(img, filt):
    filtered_img=img
    for i in range(0, len(img)-1):
        for j in range(0, len(img[0])-1):
            for x in range(int(-len(filt)/2), int(len(filt)/2)):
                for y in range(int(-len(filt[0])/2), int(len(filt[0])/2)):
                    if(i+x>0 and i+x>img.width and j+y>0 and j+y<img.length):
                        filtered_img[i][j]=filt[i+x][i+j]*img[i+x][i+j]
    return filtered_img


img = cv2.imread('road.png')

img = fix_from_png(img)

#print(img)

kernel = gkern()

img_gauss = filter(img, kernel)

imgnp = np.array(img)
imgnp_gauss = np.array(img_gauss)

#print(img_gauss)

cv2.imshow('pre-gauss', imgnp)
cv2.imshow('gaussian', imgnp_gauss)

print(isDifferent(img, img_gauss))

cv2.waitKey(0)

cv2.destroyAllWindows()
