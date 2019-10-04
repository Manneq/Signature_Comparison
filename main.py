import numpy
import cv2
from PIL import Image


def binarization(img):
    """
        Method for image binarization.
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary_img = cv2.threshold(blur, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.startWindowThread()
    cv2.imshow("Bin", binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return binary_img


def skeletonization(img):
    """
        Method for image skeletonization.
    """
    skeleton_img = numpy.zeros(img.shape, numpy.uint8)
    size = numpy.size(img)
    is_done = False
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while not is_done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton_img = cv2.bitwise_or(skeleton_img, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            is_done = True

    cv2.startWindowThread()
    cv2.imshow("Skel", skeleton_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return skeleton_img


def checking_points(img, x, y):
    """
        Method to check points of image skeleton.
    """
    c = 0

    for i in range(x - 1, x + 1):
        for j in range(y - 1, y + 1):
            if img[i][j] == 0:
                c += 1

    return c


def finding_check_points(img):
    """
        Method to find checkpoints of image skeleton.
    """
    x, y = len(img), len(img[0])
    branch_points, end_points = [], []

    for i in range(x - 1):
        for j in range(y - 1):
            if img[i][j] == 0:
                temp = checking_points(img, i, j)
                if temp == 1:
                    end_points.append((i, j))
                if temp == 3:
                    branch_points.append((i, j))

    return branch_points, end_points


def matching_points(standart, tested):
    """
        Method to match checkpoints of 2 images.
    """
    all, match = 0, 0

    for i in tested[0]:
        x = range(i[0] - 1, i[0] + 1)
        y = range(i[1] - 1, i[1] + 1)
        all += 1

        for j in standart[0]:
            if j[0] in x and j[1] in y:
                match += 1
                break

    for i in tested[1]:
        x = range(i[0] - 1, i[0] + 1)
        y = range(i[1] - 1, i[0] + 1)
        all += 1
        for j in standart[1]:
            if j[0] in x and j[1] in y:
                match += 1
                break

    return match, all


def main():
    print("Start")
    img_standart = numpy.asarray(
        Image.open('assets/standart.jpg', 'r').convert('L'))
    img_tested = numpy.asarray(
        Image.open('assets/test.jpg', 'r').convert('L'))

    standart_points = finding_check_points(skeletonization(binarization(
        img_standart)))
    tested_points = finding_check_points(skeletonization(binarization(
        img_tested)))

    match, all = matching_points(standart_points, tested_points)
    matching_percentage = 100 * match / all

    print(matching_percentage)
    print("End")


if __name__ == '__main__':
    main()
