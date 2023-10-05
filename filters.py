import cv2
import numpy as np
from scipy import ndimage


def hipass_basic(imgpath):
    img = cv2.imread(imgpath)
    hipass = img - cv2.GaussianBlur(img, (21, 21), 3)+127
    cv2.imshow("Original", img)
    cv2.imshow("Passa-Alta", hipass)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def highboost(imgpath, boost_factor):
    img = cv2.imread(imgpath)
    hipass = img - cv2.GaussianBlur(img, (7, 7), 3)+127
    hiboost = cv2.addWeighted(img, boost_factor, hipass, -1, 0)
    cv2.imshow("Original", img)
    cv2.imshow("Alto Reforço", hiboost)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def lopass_avg(imgpath, kernel):
    img = cv2.imread(imgpath)
    lopass = cv2.boxFilter(img, -1, (kernel,kernel))
    cv2.imshow('Original', img)
    cv2.imshow('Passa-Baixa', lopass)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def lopass_median(imgpath, kernel):
    img = cv2.imread(imgpath)
    lopass = cv2.medianBlur(img,kernel)
    cv2.imshow('Original', img)
    cv2.imshow('Passa-Baixa', lopass)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def grayscale(imgpath):
    img = cv2.imread(imgpath)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Original", img)
    cv2.imshow("Escala de Cinzas", grayimg)
    cv2.waitKey()
    cv2.destroyAllWindows()


def threshold(thr, imgpath):
    img = cv2.imread(imgpath)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val, thresh = cv2.threshold(grayimg, thr, 255, cv2.THRESH_BINARY)
    cv2.imshow("Original", img)
    cv2.imshow("Limiar", thresh)
    cv2.waitKey()
    cv2.destroyAllWindows()


def roberts(imgpath):
    roberts_y = np.array([[1, 0], [0, -1]])
    roberts_x = np.array([[0, 1], [-1, 0]])

    img = cv2.imread(imgpath, 0).astype('float64')
    img /= 255.0
    vertical = ndimage.convolve(img, roberts_y)
    horizontal = ndimage.convolve(img, roberts_x)

    edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
    edged_img *= 255
    edged_img = edged_img.astype(np.uint8)

    img = cv2.imread(imgpath)
    cv2.imshow("Roberts", edged_img)
    cv2.imshow("Original", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def prewitt(imgpath):
    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    prewittimg = prewittx + prewitty
    cv2.imshow("Original", img)
    cv2.imshow("Prewitt", prewittimg)
    cv2.waitKey()
    cv2.destroyAllWindows()


def canny(imgpath, low_thr, hi_thr):
    img = cv2.imread(imgpath)
    cannyimg = cv2.Canny(img, low_thr, hi_thr)
    cv2.imshow("Original", img)
    cv2.imshow("Canny", cannyimg)
    cv2.waitKey()
    cv2.destroyAllWindows()


def sobel(imgpath):
    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    img_sobelx = cv2.Sobel(img_gaussian, cv2.CV_8U, 1, 0, ksize=5)
    img_sobely = cv2.Sobel(img_gaussian, cv2.CV_8U, 0, 1, ksize=5)
    img_sobel = img_sobelx + img_sobely
    cv2.imshow("Original", img)
    cv2.imshow("Sobel", img_sobel)
    cv2.waitKey()
    cv2.destroyAllWindows()


def log(imgpath, kernel):
    image = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_image = cv2.Laplacian(image_gray, cv2.CV_16S, ksize=kernel)
    # Plot the original and filtered images
    cv2.imshow("Original", image)
    cv2.imshow("LoG", filtered_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

