import cv2
import numpy as np
import math
import os
import tkinter as tk
from tkinter import messagebox

# Set up global variables
image_files = []
current_image_index = 0
image = None
crack_image_on_bright = None
zoom_factor = 5  # Zoom factor
is_zoomed = False
mouse_x, mouse_y = 0, 0
distance_mode = False
clicked_points = []  # Store clicked points for distance measurement
roi_x1, roi_y1 = 0, 0  # Region of interest starting coordinates
small_box_size_mm = 100  # Known box size in millimeters
resize_width, resize_height = None, None  # Resizing dimensions
scaling_factor_width, scaling_factor_height = None, None  # Scaling factors

def load_folder(folder_path):
    """Load a folder and get all image files."""
    global image_files, current_image_index

    # Get all image files from the specified folder
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not image_files:
        messagebox.showerror("Error", "No image files found in the specified folder.")
        return

    # Start with the first image
    current_image_index = 0
    load_image()

def load_image():
    """Load the current image based on the index and apply all functionalities."""
    global image, crack_image_on_bright

    if current_image_index >= len(image_files):
        messagebox.showinfo("End", "All images in the folder have been processed.")
        return

    # Read the current image
    image_path = image_files[current_image_index]
    image = cv2.imread(image_path)

    if image is None:
        messagebox.showerror("Error", f"Image at path '{image_path}' could not be loaded.")
        return

    process_image()  # Process the loaded image

def perspective_correction(img, contour):
    """Apply perspective correction based on contour."""
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 4:
        pts = sorted(np.squeeze(approx), key=lambda x: (x[1], x[0]))
        pts1 = np.float32([pts[0], pts[1], pts[3], pts[2]])
        width, height = 100, 100  # Desired width and height in pixels for corrected image
        pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, matrix, (width, height))
    return None

def draw_transparent_rectangle(img, x, y, w, h, color=(0, 55, 0), alpha=0.2):
    """Draw a transparent rectangle on the image."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)  # Draw filled rectangle on overlay
    # Blend the overlay with the original image using the alpha value
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def remove_grid_lines(img):
    """Remove grid lines from the image using morphological operations."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    # Detect vertical and horizontal lines separately
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    vertical_lines = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_lines = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, horizontal_kernel)

    # Combine both vertical and horizontal lines
    grid_mask = cv2.bitwise_or(vertical_lines, horizontal_lines)

    # Dilate the mask to ensure grid lines are fully covered
    dilated_grid_mask = cv2.dilate(grid_mask, np.ones((3, 3), np.uint8), iterations=2)

    # Remove grid lines by inpainting
    img_no_grid = cv2.inpaint(img, dilated_grid_mask, 5, cv2.INPAINT_TELEA)
    
    return img_no_grid

def process_image():
    """Apply all functionalities to the current image, including scaling based on grid boxes."""
    global image, crack_image_on_bright, scaling_factor_width, scaling_factor_height

    # Step 1: Detect grid lines and calculate scaling factors based on grid boxes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    suitable_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)

        if 0.9 < aspect_ratio < 1.1 and 1000 < area < 20000:
            corrected_box = perspective_correction(gray, contour)
            if corrected_box is not None:
                suitable_contours.append((x, y, w, h, corrected_box))

    if suitable_contours:
        scaling_factors = []
        for (x, y, w, h, corrected_box) in suitable_contours:
            scaling_factor_width = small_box_size_mm / w  # mm per pixel
            scaling_factor_height = small_box_size_mm / h  # mm per pixel
            scaling_factors.append((scaling_factor_width, scaling_factor_height))
            
            # Draw transparent scaling box
            draw_transparent_rectangle(image, x, y, w, h)

        avg_scaling_factor_width = np.mean([sf[0] for sf in scaling_factors])
        avg_scaling_factor_height = np.mean([sf[1] for sf in scaling_factors])

        scaling_factor_width = avg_scaling_factor_width
        scaling_factor_height = avg_scaling_factor_height

        cv2.putText(image, f'Scaling Factor X: {scaling_factor_width:.2f} mm/px', (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f'Scaling Factor Y: {scaling_factor_height:.2f} mm/px', (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        scaling_factor_width = scaling_factor_height = None
        messagebox.showerror("Error", "No suitable scaling box found.")

    # Step 2: Create a copy of the image without grid lines for crack detection
    image_no_grid = remove_grid_lines(image)

    # Step 3: Perform crack detection on the grid-free image
    crack_image_on_bright = detect_and_measure_cracks(image_no_grid)

# Mouse callback to update mouse position and select points for distance measurement
def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y, clicked_points, is_zoomed
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        if is_zoomed:
            # Calculate the actual coordinates on the original image scale
            adjusted_x = roi_x1 + x // zoom_factor
            adjusted_y = roi_y1 + y // zoom_factor
        else:
            adjusted_x, adjusted_y = x, y

        # Add the clicked point to the list only if in distance mode
        if distance_mode:
            clicked_points.append((adjusted_x, adjusted_y))
            if len(clicked_points) > 2:
                # Keep only the last two points
                clicked_points = clicked_points[-2:]

def detect_and_measure_cracks(image_no_grid):
    global scaling_factor_width, scaling_factor_height

    gray_image = cv2.cvtColor(image_no_grid, cv2.COLOR_BGR2GRAY)
    brightened_image = cv2.convertScaleAbs(gray_image, alpha=0.87, beta=10)
    _, binary_image = cv2.threshold(brightened_image, 85, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crack_image = image_no_grid.copy()
    cv2.drawContours(crack_image, contours, -1, (0, 0, 255), 2)

    if scaling_factor_width and scaling_factor_height:
        measure_crack_widths(contours, scaling_factor_width, scaling_factor_height, crack_image)

    return crack_image

def measure_crack_widths(contours, scaling_factor_x, scaling_factor_y, crack_image):
    """Measure crack widths at regular intervals and display them with enhanced visibility."""
    for contour in contours:
        for i in range(0, len(contour), 20):
            point = contour[i][0]
            x, y = point[0], point[1]
            nearest_point = None
            min_distance = float('inf')
            for j in range(0, len(contour), 20):
                if i == j:
                    continue
                other_point = contour[j][0]
                distance = math.sqrt((other_point[0] - x) ** 2 + (other_point[1] - y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = other_point

            if nearest_point is not None:
                # Calculate crack width in mm
                dx = (nearest_point[0] - x) * scaling_factor_x
                dy = (nearest_point[1] - y) * scaling_factor_y
                crack_width_mm = math.sqrt(dx ** 2 + dy ** 2)

                # Midpoint between the two points to place the text over the crack
                mid_x = (x + nearest_point[0]) // 2
                mid_y = (y + nearest_point[1]) // 2
                
                # Draw the crack width on the image over the crack
                text_color = (0, 255, 255)  # Yellow for high contrast
                font_scale = 0.4  # Font size
                thickness = 2  # Bold text for better readability
                cv2.putText(crack_image, f"{crack_width_mm:.2f} mm", (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

def main_loop():
    """Main loop to display and interact with the images, keeping grid lines visible in output."""
    global image, crack_image_on_bright, current_image_index, is_zoomed, distance_mode, roi_x1, roi_y1, display_image

    if image is None:
        messagebox.showerror("Error", "No image loaded. Please check the folder path.")
        return

    cv2.namedWindow("Crack Detection")
    cv2.setMouseCallback("Crack Detection", on_mouse)  # Set mouse callback to track cursor position

    while True:
        h, w = image.shape[:2]

        # Overlay the crack detection output on the original image
        if crack_image_on_bright is not None:
            display_image = cv2.addWeighted(image, 0.7, crack_image_on_bright, 0.3, 0)
        else:
            display_image = image.copy()

        # Handle zoomed view if is_zoomed is True
        if is_zoomed:
            zoom_x1 = max(0, min(mouse_x - w // (2 * zoom_factor), w - w // zoom_factor))
            zoom_y1 = max(0, min(mouse_y - h // (2 * zoom_factor), h - h // zoom_factor))
            roi_x1, roi_y1 = zoom_x1, zoom_y1

            # Crop and resize the zoomed region
            zoomed_image = display_image[zoom_y1:zoom_y1 + h // zoom_factor, zoom_x1:zoom_x1 + w // zoom_factor]
            zoomed_image = cv2.resize(zoomed_image, (w, h), interpolation=cv2.INTER_LINEAR)
            display_image = zoomed_image.copy()  # Use a copy to allow drawing without affecting zoomed_image

        # Draw clicked points, line, and calculate/display distance if two points are selected
        if len(clicked_points) == 2:
            pt1, pt2 = clicked_points
            if is_zoomed:
                # Adjust point coordinates for zoomed view
                pt1_display = ((pt1[0] - roi_x1) * zoom_factor, (pt1[1] - roi_y1) * zoom_factor)
                pt2_display = ((pt2[0] - roi_x1) * zoom_factor, (pt2[1] - roi_y1) * zoom_factor)
            else:
                pt1_display, pt2_display = pt1, pt2

            # Draw points and connecting line
            cv2.circle(display_image, pt1_display, 5, (0, 0, 255), -1)
            cv2.circle(display_image, pt2_display, 5, (0, 0, 255), -1)
            cv2.line(display_image, pt1_display, pt2_display, (255, 255, 0), 2)

            # Calculate the distance using the original coordinates
            dx = (pt2[0] - pt1[0]) * scaling_factor_width
            dy = (pt2[1] - pt1[1]) * scaling_factor_height
            distance_mm = math.sqrt(dx ** 2 + dy ** 2)

            # Display the distance at the midpoint between points
            mid_x = (pt1_display[0] + pt2_display[0]) // 2
            mid_y = (pt1_display[1] + pt2_display[1]) // 2
            cv2.putText(display_image, f"{distance_mm:.2f} mm", (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)  # Yellow text for distance

        cv2.imshow("Crack Detection", display_image)

        # Wait for a key event
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):  # Next image
            current_image_index += 1
            load_image()
        elif key == ord('b'):  # Previous image
            current_image_index = max(0, current_image_index - 1)
            load_image()
        elif key == ord('m'):  # Toggle zoom
            is_zoomed = not is_zoomed
        elif key == ord('d'):  # Toggle distance mode
            distance_mode = not distance_mode
        elif key == ord('q'):  # Quit
            break

    cv2.destroyAllWindows()

# Ensure a valid folder path before starting
folder_path = r"E:\New Folder"  # Update this with the actual folder path
if os.path.isdir(folder_path):
    load_folder(folder_path)
    main_loop()
else:
    messagebox.showerror("Error", "Folder path does not exist.")
