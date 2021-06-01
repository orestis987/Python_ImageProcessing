import numpy as np
import cv2

def valid(x):
    if x==None:
        print("variable not set")


def dummy(val):
    pass



identity_kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
gaussian_kernel1 = cv2.getGaussianKernel(3,0)
gaussian_kernel2 = cv2.getGaussianKernel(5,0)
box_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]], np.float32)/9.0


kernels = [identity_kernel, sharpen_kernel, gaussian_kernel1, gaussian_kernel2, box_kernel]

cv2.namedWindow("MyApp")
img_original = cv2.imread("myimage.jpg")
img_modify = img_original.copy()

img_gray_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
img_gray_modify = img_gray_original.copy()

cv2.createTrackbar("contrast", "MyApp", 1, 100, dummy)
cv2.createTrackbar("brightness", "MyApp", 50, 100, dummy)
cv2.createTrackbar("filter", "MyApp", 0, len(kernels)-1, dummy)
cv2.createTrackbar("grayscale", "MyApp", 0, 1, dummy)

count = 1

while True:
    grayscale = cv2.getTrackbarPos("grayscale", "MyApp")
    if grayscale == 0:
        cv2.imshow("MyApp", img_modify)
    else:
        cv2.imshow("MyApp", img_gray_modify)


    k = cv2.waitKey(1) & 0xFF #?
    if k == ord('q'): # takes the asci code of q
        break
    elif k == ord('s'):
        if grayscale == 0:
            cv2.imwrite('output%d.png' %count, img_modify)
        else:
            cv2.imwrite('output%d.png' %count, img_gray_modify)
        count += 1

    contrast = cv2.getTrackbarPos("contrast", "MyApp")
    brightness = cv2.getTrackbarPos("brightness", "MyApp")
    filter = cv2.getTrackbarPos("filter", "MyApp")

    img_modify = cv2.filter2D(img_original, -1, kernels[filter])
    img_modify = cv2.addWeighted(img_modify, contrast, np.zeros(img_original.shape, dtype=img_original.dtype), 0, brightness-50) #img (I1), contrast, I2, b, G (I'= a*I1 + bI2 + G)

    img_gray_modify = cv2.filter2D(img_gray_original, -1, kernels[filter])
    img_gray_modify = cv2.addWeighted(img_gray_modify, contrast, np.zeros(img_gray_original.shape, dtype=img_gray_original.dtype), 0, brightness-50) #img (I1), contrast, I2, b, G (I'= a*I1 + bI2 + G)

cv2.destroyAllWindows()



"""
img = cv2.imread("myimage.jpg")
valid(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cb_img = cv2.addWeighted(img, 4, np.zeros(img.shape, dtype=img.dtype), 0, 100) #img (I1), contrast, I2, b, G (I'= a*I1 + bI2 + G)

#print img.shape

K = np.array([
    [0, -1, 0],
    [-1, 5,-1],
    [0, -1, 0]
])#sharpening

convolved = cv2.filter2D(img, -1, K)

cv2.imshow("myWindowName",img)
#cv2.imshow("myWindowNameGray",gray)
#cv2.imshow("myWindowNameContrast",cb_img)
cv2.imshow("myWindowNameConvolved",convolved)
cv2.waitKey(0) # take how long it will wait, or 0 for ever until pressing a key
cv2.destroyAllWindows()
"""