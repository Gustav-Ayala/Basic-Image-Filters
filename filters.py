import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
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
    hipass = img - cv2.blur(img, (7, 7))+127
    hiboost = cv2.addWeighted(img, boost_factor+1, hipass, -boost_factor, 0)
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


def sobel(imgpath, kernel):
    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    img_sobelx = cv2.Sobel(img_gaussian, cv2.CV_8U, 1, 0, ksize=kernel)
    img_sobely = cv2.Sobel(img_gaussian, cv2.CV_8U, 0, 1, ksize=kernel)
    img_sobel = img_sobelx + img_sobely
    cv2.imshow("Original", img)
    cv2.imshow("Sobel", img_sobel)
    cv2.waitKey()
    cv2.destroyAllWindows()


def log(imgpath, kernel):
    image = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    log = cv2.Laplacian(image_gray, cv2.CV_16S, ksize=kernel)
    cv2.imshow("Original", image)
    cv2.imshow("LoG", log)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return log


def zerocross(imgpath, kernel):
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    imgblur = cv2.GaussianBlur(img, (3, 3), 0)
    imggray = cv2.cvtColor(imgblur, cv2.COLOR_BGR2GRAY)
    log = cv2.Laplacian(imggray, cv2.CV_16S, ksize=kernel)
    mat = np.zeros(log.shape)

    for i in range(0, log.shape[0] - 1):
        for j in range(0, log.shape[1] - 1):
            if log[i][j] > 0:
                if log[i + 1][j] < 0 or log[i + 1][j + 1] < 0 or log[i][j + 1] < 0:
                    mat[i, j] = 1
            elif log[i][j] < 0:
                if log[i + 1][j] > 0 or log[i + 1][j + 1] > 0 or log[i][j + 1] > 0:
                    mat[i, j] = 1
    cv2.imshow("Zerocross", mat)
    cv2.imshow("Original", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
def watershed(imgpath):
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    originalimg = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    cv2.watershed(img, markers)
    img[markers == -1] = [0, 0, 255]
    cv2.imshow("Original", originalimg)
    cv2.imshow("Watershed", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def histogram(imgpath):
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([imggray], [0], None, [256], [0,256])

    plt.bar(range(256), hist.ravel(), width=1, color='magenta')
    plt.title('Histograma da Imagem (Grayscale)')
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')
    plt.show()
    cv2.imshow("Image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def histogram_adapt(imgpath):
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img)
    cv2.imshow("Image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def count(imgpath):
    image = cv2.imread(imgpath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    canny = cv2.Canny(blur, 30, 150, 3)
    dilated = cv2.dilate(canny, (1, 1), iterations=0)
    (cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
    text = "{} objetos".format(len(cnt))
    cv2.putText(rgb, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 0, 159), 2)
    cv2.imshow("Image", rgb)
    cv2.waitKey()
    cv2.destroyAllWindows()


def noise(imgpath, amount):
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    num_salt = np.ceil(amount * imggray.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in imggray.shape]
    imggray[coords[0], coords[1]] = 255

    num_pepper = np.ceil(amount * imggray.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in imggray.shape]
    imggray[coords[0], coords[1]] = 0
    cv2.imshow("Original", img)
    cv2.imshow("Salt and Pepper", imggray)

    cv2.waitKey()
    cv2.destroyAllWindows()

