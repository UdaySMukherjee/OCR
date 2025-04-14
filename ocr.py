import numpy as np
import cv2
import imutils
from pyzbar.pyzbar import decode

print("I am Atrij")

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
