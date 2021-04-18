import cv2
import os

cascades_dir = "/home/wojtek/anaconda3/envs/gpu/share/opencv4/haarcascades"
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # read video frames

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        os.path.join(cascades_dir, "haarcascade_frontalface_alt.xml"))
    eye_cascade = cv2.CascadeClassifier(
        os.path.join(cascades_dir, "haarcascade_eye.xml"))
    smile_cascade = cv2.CascadeClassifier(
        os.path.join(cascades_dir, "haarcascade_smile.xml"))

    faces = face_cascade.detectMultiScale(gray, 1.1, minNeighbors=5)
    eyes = eye_cascade.detectMultiScale(gray, 1.05, minNeighbors=55)
    smiles = smile_cascade.detectMultiScale(
        gray, 1.15, minNeighbors=250, maxSize=(200, 140))

    for (x, y, width, height) in faces:
        cv2.rectangle(gray, pt1=(x, y), pt2=(x+width, y+height),
                      color=(128, 128, 128), thickness=8)

    for (x, y, width, height) in eyes:
        cv2.circle(gray, center=((x+width//2), (y+height//2)),
                   radius=width//2, color=(0, 255, 0), thickness=8)

    for (x, y, width, height) in smiles:
        cv2.rectangle(gray, pt1=(x, y), pt2=(x+width, y+height),
                      color=(255, 0, 0), thickness=8)

    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
