import cv2
from random import randrange
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# img=cv2.imread('det2.jpeg')
webcam=cv2.VideoCapture(0)

while True:
    successful_frame_read, frame=webcam.read()
    greyscaled_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_coordinates=trained_face_data.detectMultiScale(greyscaled_frame)
    # print(face_coordinates)
    # for i in face_coordinates:
    #      cv2.rectangle(frame,(i[0],i[1]),(i[0]+i[2],i[3]+i[1]),(255,0,0),4)
    cv2.imshow('sadf',frame)
    print(face_coordinates)
    if len(face_coordinates)>1:
        print("Another person detected")
        break
    key=cv2.waitKey(1)
    if key==81 or key==113:
        break
# cv2.imshow('First python detection!',img)
webcam.release()
print("Code Completed")