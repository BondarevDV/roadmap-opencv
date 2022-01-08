import cv2
import datetime
import os
from time import gmtime, strftime
from pathlib import Path, PurePath
import os, shutil

PATH_DATA = "../data/"


def clear_data(folder=PATH_DATA):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    


def blur_face(img):
    print(type(img))
    (h, w) = img.shape[:2]
    dW = int(w)
    dH = int(h)
    if dW % 2 == 0:
        dW -= 1
    if dH % 2 == 0:
        dH -= 1
    
    return cv2.GaussianBlur(img, (dW, dH), 0)




def main():
    capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade-class/haarcascade_frontalface_default.xml')

    while True:
        ret, img = capture.read()
        PurePath(PATH_DATA)
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=5, minSize=(28, 28))
        
        for (x, y, w, h) in faces:
            path = PurePath(PATH_DATA, f'{strftime("%Y-%m-%d%H%M%S", gmtime())}.png')
            isWritten = cv2.imwrite(str(path), img)
            blur_face(img)
            if isWritten:
                print('Image is successfully saved as file.')
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0),2)
            img[y: y + h, x: x + w] = blur_face(img[y: y + h, x: x + w])
                
        cv2.imshow('From Camera', img)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    clear_data()
    main()