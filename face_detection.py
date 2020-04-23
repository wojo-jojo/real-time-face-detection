import cv2
import matplotlib.pyplot as plt

sample_img_path = "images/standing-family-near-fireplace-1648387.jpg"
sample_img = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)
# grayscale_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
plt.imshow(sample_img)
plt.show()
