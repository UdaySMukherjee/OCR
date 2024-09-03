import numpy as np
import cv2
import imutils
from pyzbar.pyzbar import decode

def detect(image):
    # Check if the image is loaded
    if image is None:
        print("Error: Image not loaded.")
        return None
    
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
    box = np.array(box, dtype=np.int32)
    
    # Return the bounding box of the detected region
    return box

def BarcodeReader(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Check if the image is loaded
    if img is None:
        print(f"Error: Unable to load image from path {image_path}")
        return
    
    # Detect the bounding box
    box = detect(img)
    
    if box is not None:
        # Draw the bounding box
        cv2.polylines(img, [box], True, (0, 255, 0), 2)
    
        # Decode the barcode image
        detectedBarcodes = decode(img)
    
        if not detectedBarcodes:
            print("Barcode Not Detected or your barcode is blank/corrupted!")
        else:
            # Traverse through all the detected barcodes in image
            for barcode in detectedBarcodes:
                # Locate the barcode position in image
                (x, y, w, h) = barcode.rect
                
                # Put the rectangle in image using cv2 to highlight the barcode
                cv2.rectangle(img, (x-10, y-10), (x + w+10, y + h+10), (255, 0, 0), 2)
                
                # Print the barcode data
                print("Barcode Data:", barcode.data.decode('utf-8'))
                print("Barcode Type:", barcode.type)
    
        # Display the image
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No barcode region detected.")

if __name__ == "__main__":
    # Path to the image
    image_path = r"data/1.jpg"
    BarcodeReader(image_path)
