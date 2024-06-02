### README

# Face Detection with Unique Bounding Boxes

This repository contains scripts for detecting faces in images and video streams using OpenCV and drawing unique bounding boxes around detected faces. The unique bounding boxes are created using a combination of lines and ellipses to give a more distinctive appearance.

## Requirements

- Python 3.x
- OpenCV
- Matplotlib (for displaying images)

## Installation

1. Install Python 3.x from the [official website](https://www.python.org/downloads/).
2. Install the required libraries using pip:

```bash
pip install opencv-python
pip install opencv-contrib-python
pip install matplotlib
```

## Usage

### Script 1: Face Detection in Images

This script reads an image, detects faces, and draws unique bounding boxes around them.

#### Code

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Path to the Haar cascade file for face detection
cascPath = "haarcascade_frontalface_default.xml"
# Create the Haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread("facess.png")
# Convert the image to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,  # Parameter specifying how much the image size is reduced at each image scale
    minNeighbors=5,  # Parameter specifying how many neighbors each candidate rectangle should have to retain it
    minSize=(30, 30)  # Minimum possible object size. Objects smaller than this are ignored.
)

print("Found {0} faces!".format(len(faces)))

# Function to draw a unique bounding box around detected faces
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left corner
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right corner
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left corner
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right corner
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

# Draw bounding boxes around all detected faces
for (x, y, w, h) in faces:
    draw_border(image, (x, y), (x + w, y + h), (0, 255, 0), 2, 10, 20)

# Display the image with bounding boxes
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axis
plt.show()
```

#### Output:
![image](https://github.com/MuhammadRaheelNaseem/Face-Detection-with-Unique-Bounding-Boxes/assets/63813881/5d2d2f5f-95aa-4b50-90b1-7fe41503959e)


### Script 2: Face Detection in Video

This script captures video from a webcam, detects faces in real-time, and draws unique bounding boxes around them.

#### Code

```python
import cv2

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Function to draw a unique bounding box around detected faces
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left corner
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right corner
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left corner
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right corner
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

while True:
    # Read the frame from the webcam
    _, img = cap.read()
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw bounding boxes around all detected faces
    for (x, y, w, h) in faces:
        draw_border(img, (x, y), (x + w, y + h), (0, 255, 0), 2, 10, 20)

    # Display the frame with bounding boxes
    cv2.imshow('Video', img)
    
    # Stop if the escape key is pressed
    if cv2.waitKey(30) & 0xff == 27:
        break

# Release the VideoCapture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
```

## Explanation

- **Face Detection:** Both scripts use OpenCV's Haar cascades to detect faces in images and video streams.
- **Unique Bounding Boxes:** The `draw_border` function creates a unique bounding box by combining lines and ellipses at the corners.
- **Real-time Video:** The second script captures video from the webcam and processes each frame in real-time to detect faces and draw bounding boxes.

## Running the Scripts

### Script 1

1. Save the script as `face_detection_image.py`.
2. Place your image file (`facess.png`) in the same directory.
3. Ensure `haarcascade_frontalface_default.xml` is in the same directory or provide the correct path.
4. Run the script:

```bash
python face_detection_image.py
```

### Script 2

1. Save the script as `face_detection_video.py`.
2. Ensure `haarcascade_frontalface_default.xml` is in the same directory or provide the correct path.
3. Run the script:

```bash
python face_detection_video.py
```

## Customization

- Adjust `scaleFactor`, `minNeighbors`, and `minSize` parameters in `detectMultiScale` for better face detection results based on your use case.
- Modify the `draw_border` function to customize the appearance of the bounding boxes.
