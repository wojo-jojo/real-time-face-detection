import cv2
import os
import random


# File paths
img_dir = os.path.abspath("../images")
image_samples = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir)]
cascades_dir = "/home/wojtek/anaconda3/envs/gpu/share/opencv4/haarcascades"

# Pick sample image
sample_img = cv2.imread(image_samples[random.randint(0, len(image_samples) - 1)])
gray_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(
    os.path.join(cascades_dir, "haarcascade_frontalface_alt.xml"))
eye_cascade = cv2.CascadeClassifier(
    os.path.join(cascades_dir, "haarcascade_eye.xml"))
smile_cascade = cv2.CascadeClassifier(
    os.path.join(cascades_dir, "haarcascade_smile.xml"))

faces = face_cascade.detectMultiScale(sample_img, 1.1, minNeighbors=5)
eyes = eye_cascade.detectMultiScale(sample_img, 1.05, minNeighbors=55)
smiles = smile_cascade.detectMultiScale(
    sample_img, 1.15, minNeighbors=250, maxSize=(200, 140))

for (x, y, width, height) in faces:
    cv2.rectangle(sample_img, pt1=(x, y), pt2=(x+width, y+height),
                  color=(0, 0, 255), thickness=8)

for (x, y, width, height) in eyes:
    cv2.circle(sample_img, center=((x + width//2), (y+height//2)), radius=width//2,
               color=(0, 255, 0), thickness=8)

for (x, y, width, height) in smiles:
    cv2.rectangle(sample_img, pt1=(x, y), pt2=(x+width, y+height),
                  color=(255, 0, 0), thickness=8)


detect_count = {
    "faces": 0,
    "eyes": 0,
    "smiles": 0
}
map_detectors = {
    "faces": faces,
    "eyes": eyes,
    "smiles": smiles
}

for feature in detect_count.keys():
    try:
        detect_count[feature] += map_detectors[feature].shape[0]
    except AttributeError:
        pass  # feature not found in image

print("Found:", )
for feature, count in detect_count.items():
    print(count, feature)

small_img = cv2.resize(sample_img, (0, 0), fx=0.5, fy=0.5)
cv2.imshow("Sample image", small_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

save_img = input("Save image? (y/n) ")
if save_img.lower() == "y":
    filename = input("Enter filename: ")
    cv2.imwrite(os.path.join("../", "demo_img", filename + ".jpg"), small_img)
else:
    pass
