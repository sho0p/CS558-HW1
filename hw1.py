from PIL import Image
import cv2
from scipy.signal import sepfir2d
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


'''returns a 2d gaussian kernel'''
def gkern(len=5, nsig=1):
    # ax = np.arange(-len // 2+1.//2+1.)
    # xx,yy = np.meshgrid(ax,ax)

    # kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(nsig))

    # return kernel/np.sum(kernel)

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

def filter(img, filt):
    filtered_img=img
    height = len(img)
    width = len(img[0])
    for i in range(0, height-1):
        for j in range(0, width-1):
            for x in range(-len(filt)//2, len(filt)//2):
                for y in range(-len(filt[0])//2, len(filt[0])//2):
                    if(i+y>0 and i+y>height and j+x>0 and j+x<height):
                        filtered_img[i][j]=filt[i+x][i+j]*img[i+x][i+j]
    return filtered_img


img = cv2.imread('road.png')

img = fix_from_png(img)

#print(img)

kernel = gkern()*25
print(kernel)

img_gauss = filter(img, kernel)

imgnp = np.array(img)
imgnp_gauss = np.array(img_gauss)

#print(img_gauss)

cv2.imshow('pre-gauss', imgnp)
cv2.imshow('gaussian', imgnp_gauss)

print(img == img_gauss)

cv2.waitKey(0)

cv2.destroyAllWindows()
