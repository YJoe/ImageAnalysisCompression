import numpy
import cv2

cv2.namedWindow("a duck", cv2.WINDOW_NORMAL)
duck_image = cv2.imread("C:\duck-05.jpg")
cv2.imshow("a duck", duck_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
