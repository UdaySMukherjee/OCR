import numpy as np
import cv2
import imutils

def detect(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute the gradient magnitude representation of the image
    ddepth = cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    
    # Subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    
    # Blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    
    # Construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    
    # Find the contours in the thresholded image
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # If no contours were found, return None
    if len(cnts) == 0:
        return None
    
    # Otherwise, sort the contours by area and compute the rotated bounding box of the largest contour
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.int32)  # Use np.int32 for conversion
    
    # Return the bounding box of the detected region
    return box

# Load the image
img = cv2.imread(r'data\2007002124862-01_N95-2592x1944_scaledTo640x480bilinear_jpg.rf.c8c0764023808e4b13d21ef84e841b8e.jpg')

# Detect the bounding box
box = detect(img)

# If a bounding box was detected
if box is not None:
    # Draw the bounding box
    cv2.polylines(img, [box], True, (0, 255, 0), 2)
    
    # Optionally, put text or any additional information
    text_position = (int(box[0][0]), int(box[0][1]))  # Position for text
    cv2.putText(img, 'Detected Barcode', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Show the result
    cv2.imshow('Detected Barcode', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No barcode detected.")
