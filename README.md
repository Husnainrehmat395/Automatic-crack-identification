# Automatic-crack-identification
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import simpledialog, Tk, filedialog

# Initialize Tkinter to get GUI for user input
root = Tk()
root.withdraw()

# Ask the user for the folder containing the images
folder_path = filedialog.askdirectory(title="Select Folder Containing Crack Images")
if not folder_path:
    raise FileNotFoundError("No folder selected. Please select a valid folder containing images.")

# Ask the user for real-world distances in the X-axis and Y-axis
real_world_width = simpledialog.askfloat("Input", "Real-world width of the images (X-axis) in mm:")
real_world_height = simpledialog.askfloat("Input", "Real-world height of the images (Y-axis) in mm:")

if real_world_width is None or real_world_height is None:
    raise ValueError("Real-world dimensions are required for calibration.")

# Process each image in the selected folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(folder_path, filename)

        # Load the image
        image = cv2.imread(image_path)

        # If the image fails to load, skip to the next file
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get image dimensions (width and height in pixels)
        img_height, img_width = gray_image.shape

        # Calculate scaling factors for X and Y axes
        scaling_factor_x = real_world_width / img_width  # mm per pixel in X-axis
        scaling_factor_y = real_world_height / img_height  # mm per pixel in Y-axis

        print(f"\nProcessing: {filename}")
        print(f"Image Width: {img_width}px, Real-World Width: {real_world_width}mm, Scaling Factor X: {scaling_factor_x:.2f}mm/pixel")
        print(f"Image Height: {img_height}px, Real-World Height: {real_world_height}mm, Scaling Factor Y: {scaling_factor_y:.2f}mm/pixel")

        # Brighten the grayscale image by increasing pixel values
        brightened_gray_image = cv2.convertScaleAbs(gray_image, alpha=1.0, beta=0)  # Double the brightness

        # Apply thresholding to isolate cracks on the brightened image
        _, binary_image = cv2.threshold(brightened_gray_image, 100, 255, cv2.THRESH_BINARY_INV)

        # Use contours to detect cracks
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the brightened grayscale image to draw contours on
        crack_image_on_bright = cv2.cvtColor(brightened_gray_image, cv2.COLOR_GRAY2BGR)

        # Draw the contours on the brightened grayscale image with enhanced color and thickness
        cv2.drawContours(crack_image_on_bright, contours, -1, (0, 0, 255), 2)  # Red color with thickness 2

        # Calculate thickness of cracks by analyzing contours
        crack_thickness_pixels = []
        crack_thickness_mm = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                # Estimate thickness as area-to-perimeter ratio (in pixels)
                thickness_pixels = area / perimeter
                crack_thickness_pixels.append(thickness_pixels)

                # Convert thickness to real-world units (in mm)
                thickness_mm = thickness_pixels * np.sqrt(scaling_factor_x * scaling_factor_y)
                crack_thickness_mm.append(thickness_mm)

        # Display results for the current image
        print(f'Estimated crack thicknesses (in pixels): {crack_thickness_pixels}')
        print(f'Estimated crack thicknesses (in mm): {crack_thickness_mm}')

        # Resize the detected crack image for separate display
        new_width = 800
        new_height = int(new_width * crack_image_on_bright.shape[0] / crack_image_on_bright.shape[1])
        resized_crack_image = cv2.resize(crack_image_on_bright, (new_width, new_height))

        # Display the resized crack detection image in a separate window
        cv2.imshow(f'Detected Crack Image: {filename}', resized_crack_image)

        # Plot the original and processed images
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        # Grayscale image
        plt.subplot(1, 4, 2)
        plt.imshow(gray_image, cmap='gray')
        plt.title('Grayscale Image')

        # Brightened grayscale image
        plt.subplot(1, 4, 3)
        plt.imshow(brightened_gray_image, cmap='gray')
        plt.title('Brightened Grayscale Image')

        # Crack detection on brightened grayscale image
        plt.subplot(1, 4, 4)
        plt.imshow(cv2.cvtColor(crack_image_on_bright, cv2.COLOR_BGR2RGB))
        plt.title('Crack Detection on Brightened Image')

        plt.show()

        # Wait for a key press to close the displayed windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()
