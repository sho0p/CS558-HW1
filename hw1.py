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
    new_filt = []
    for row in filt:
        new_row = []
        for item in row:
            new_row.append(item/div)
        new_filt.append(new_row)
    return new_filt

def gaussian_filt(img, filt):
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
                            if new_val < -255:
                                new_val = -255
                            if new_val > 255:
                                new_val = 255
                            filtered_img[i][j] = new_val
                        except IndexError:
                            print("ERROR: i = ", i, " j = ", j, " x = ", x+half, " y = ", y+half, " half = ", half)
    return filtered_img

def sobel(img, kernx, kerny):
    rows = len(img)
    col = len(img[0])
    mag = np.zeros((rows, col))
    S1 = np.zeros((rows, col))
    S2 = np.zeros((rows, col))


    for i in range(1, rows-2):
        for j in range(1, col-2):
            S1[i][j] += kernx[0][0]*img[i-1][j-1]
            S2[i][j] += kerny[0][0]*img[i-1][j-1]

            S1[i][j] += kernx[0][1]*img[i-1][j]
            S2[i][j] += kerny[0][1]*img[i-1][j]

            S1[i][j] += kernx[0][2]*img[i-1][j+1]
            S2[i][j] += kerny[0][2]*img[i-1][j+1]

            S1[i][j] += kernx[1][0]*img[i][j-1]
            S2[i][j] += kerny[1][0]*img[i][j-1]

            S1[i][j] += kernx[1][1]*img[i][j]
            S2[i][j] += kerny[1][1]*img[i][j]

            S1[i][j] += kernx[1][2]*img[i][j+1]
            S2[i][j] += kerny[1][2]*img[i][j+1]

            S1[i][j] += kernx[2][0]*img[i+1][j-1]
            S2[i][j] += kerny[2][0]*img[i+1][j-1]

            S1[i][j] += kernx[2][1]*img[i+1][j]
            S2[i][j] += kerny[2][1]*img[i+1][j]

            S1[i][j] += kernx[2][2]*img[i+1][j+1]
            S2[i][j] += kerny[2][2]*img[i+1][j+1]

            mag[i+1][j+1] = math.sqrt(S1[i][j]**2 + S2[i][j]**2)
    
    return mag


def hessian(img):
    imgnd = np.asarray(img)
    img_grad = np.gradient(img)
    hess = np.empty((imgnd.ndim, imgnd.ndim) + imgnd.shape, dtype=imgnd.dtype)
    for k, grad_k in enumerate(img_grad):
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hess[k,l,:,:] = grad_kl
    return hess


def percentSame(img1, img2):
    height = len(img1)
    width = len(img1[0])
    div_factor = float(width * height)
    cnt = 0
    for i in range(height):
        for j in range(width) :
            if img1[i][j] == img2[i][j]:
                cnt=cnt+1
    return cnt/div_factor * 100.0

def sobel_fix(sobel):
    for row in sobel:
        for member in row:
            member = abs(member) 

'''
    takes in the 4x4 matrix that contains all hessian data
    returns hessian determinant values for each pixel
'''
def hesdet(hes):
    row = len(hes[0][0])
    print(row)
    col = len(hes[0][0][0])
    print(col)
    hesd = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            try:
                hesd = hes[0][0][i][j]* hes[1][1][i][j]-hes[0][1]*hes[1][0]
            except KeyboardInterrupt:
                print("Oh boy: i = ", i, " j = ", j)
    return hesd


def nonmaxsup(hes):
#    print(np.asarray(hes_g).shape)
    row = len(hes)
    col = len(hes[0])
    output = np.zeros((row, col))
   #print(hes_g[0][34][433]," ",hes_g[1][34][433])
    for i in range(1,row-1):
        for j in range(1,col-1):
            neighbor =  [hes[i-1][j-1], hes[i-1][j], hes[i-1][j+1] ,
                        hes[i][j-1], hes[i][j], hes[i][j+1],
                        hes[i+1][j-1], hes[i+1][j], hes[i+1][j+1]]
            if (max(neighbor) == hes[i][j] and hes[i][j] != 0):
                output[i][j] = 255
    return output

def localize_hess(hes, kern):
    row = len(hes[0][0])
    print(row)
    col = len(hes[0][0][0])
    print(col)
    half = math.floor(len(kern)/2)
    hes_new = np.zeros((len(hes),len(hes[:]),row,col))
    print(kern)
    print(half)
    for i in range(half, row-half):
        for j in range(half, col-half):
            for x in range(-half, half):
                for y in range(-half, half):
                    hes_new[0][0][i][j] += hes[0][0][i+x][j+y]*kern[x+half][y+half]
                    hes_new[0][1][i][j] += hes[0][1][i+x][j+y]*kern[x+half][y+half]
                    hes_new[1][0][i][j] += hes[1][0][i+x][j+y]*kern[x+half][y+half]
                    hes_new[1][1][i][j] += hes[1][1][i+x][j+y]*kern[x+half][y+half]
    return hes_new
        

def sobel_companion(sobel, thresh=70):
    height = len(sobel)
    width = len(sobel[0])
    for i in range(height):
        for j in range(width):
            if(sobel[i][j] <= thresh):
                new_sobel = 0
            else:
                new_sobel = sobel[i][j]
    return new_sobel



def toImage(img):
    img_arr = np.asarray(img)
    return Image.fromarray(img_arr.astype('uint8'))


img = cv2.imread('road.png')

img = fix_from_png(img)

kernel = gkern()
print(kernel)

img_gauss = gaussian_filt(img, kernel)

img_arr = np.asarray(img)
img_gauss_arr = np.asarray(img_gauss)

ime = Image.fromarray(img_arr)
ime_guass = Image.fromarray(img_gauss_arr)

ime.show()
ime_guass.show()

sobelkernx = sobelkernx_base#divFilt(sobelkernx_base, 1.0)
sobelkerny = sobelkerny_base# divFilt(sobelkerny_base, 1.0)

print(sobelkernx)
print(sobelkerny)

sobelimg = sobel(img_gauss, sobelkernx, sobelkerny)

toImage(sobelimg).show()

hes = hessian(sobelimg)
print(hes)
print(hes.shape)

hesl=localize_hess(hes, gkern(len=3))

print(hesl.shape)

hesd = hesdet(hesl)


hesdcop = copy.deepcopy(hesd)
print(hesd)
print(hesd.shape)
#   thresh = float(input("Please input threshold value: "))
thresh = 12.6
super_threshold_indices = abs(hesd) < thresh
hesd[super_threshold_indices] = 0
super_threshold_indices = abs(hesd) > 0
hesd[super_threshold_indices] = 255
#    toImage(hesd).show()
hesg = nonmaxsup(hesd)
#    print(hesg)
toImage(hesg).show()
hesd = copy.deepcopy(hesdcop)



#toImage(sobelx).show()
#toImage(sobely).show()

#print(percentSame(sobelx, sobely))

#ime_guass.show()

#cv2.waitKey(0)

#cv2.destroyAllWindows()
