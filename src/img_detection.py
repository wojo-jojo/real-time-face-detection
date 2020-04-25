import cv2
import os

img_dir = "../images"
image_samples = os.listdir(img_dir)
cascades_dir = "/home/wojtek/anaconda3/envs/gpu/share/opencv4/haarcascades"

# Pick sample image or specify own path
sample_img = cv2.imread(os.path.join(img_dir, image_samples[0]))
gray_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(
    os.path.join(cascades_dir, "haarcascade_frontalface_alt.xml"))
eye_cascade = cv2.CascadeClassifier(
    os.path.join(cascades_dir, "haarcascade_eye.xml"))
smile_cascade = cv2.CascadeClassifier(
    os.path.join(cascades_dir, "haarcascade_smile.xml"))

faces_detected = face_cascade.detectMultiScale(sample_img, 1.1, minNeighbors=5)
eyes_detected = eye_cascade.detectMultiScale(sample_img, 1.05, minNeighbors=55)
smiles_detected = smile_cascade.detectMultiScale(
    sample_img, 1.15, minNeighbors=250, maxSize=(200, 140))

detect_count = {
    "faces": 0,
    "eyes": 0,
    "smiles": 0
}
map_detectors = {
    "faces": faces_detected,
    "eyes": eyes_detected,
    "smiles": smiles_detected
}
for feature in detect_count.keys():
    try:
        detect_count[feature] += map_detectors[feature].shape[0]
    except AttributeError:
        pass  # feature not found in image

print("Found:", )
for feature, count in detect_count.items():
    print(count, feature)

for (x, y, width, height) in faces_detected:
    cv2.rectangle(sample_img, (x, y), (x + width, y + height), (0, 0, 255), 8)

for (x, y, width, height) in eyes_detected:
    cv2.circle(sample_img, ((x+width//2), (y+height//2)), width//2, (0, 255, 0), 8)

for (x, y, width, height) in smiles_detected:
    cv2. rectangle(sample_img, (x, y), (x+width, y+height), (255, 0, 0), 8)

small_img = cv2.resize(sample_img, (0, 0), fx=0.5, fy=0.5)
cv2.imshow("Sample image", small_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

save_img = input("Save image? (y/n) ")
if save_img.lower() == "y":
    filename = input("Enter filename: ")
    cv2.imwrite(os.path.join("../", filename), small_img)
else:
    pass
