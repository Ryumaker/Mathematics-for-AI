import glob
import cv2

S=set()
for i,j in enumerate(glob.glob("./dataset/tarball-lite-master/AFAD-Lite/*/*/*.jpg")):
    if i==5000:
        break
    img = cv2.imread(j)
    res = cv2.resize(img,(256,256))
    cv2.imwrite("dataset/img256/" + str(i)+".png", res)

