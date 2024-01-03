import cv2
import csv

data = {}
file_path = 'ground_truth.csv'

# Define color ranges in HSV format for apple detection.
low_apple_red = (160.0, 153.0, 153.0)
high_apple_red = (180.0, 255.0, 255.0)
low_apple_raw = (0.0, 150.0, 150.0)
high_apple_raw = (15.0, 255.0, 255.0)
low_apple_green = (25.0, 100.0, 100.0)
high_apple_green = (70.0, 255.0, 255.0)

# Read ground truth data from a CSV file.
with open(file_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)

    # Iterate through each row of the CSV file.
    for row in csv_reader:
        # First column is the image file name, second column is the apple count.
        image_name = row[0]
        apple_count = int(row[1])

        # Store data in a dictionary for later use.
        data[image_name] = apple_count

corret_num = 0
total_num = len(data)

# Process each image in the dataset.
for image_name, apple_count in data.items():

    image_path = 'counting/images/' + image_name

    # Read the image using OpenCV.
    image_bgr = cv2.imread(image_path)
    image = image_bgr.copy()
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Apply color range filters to detect apples of different colors.
    mask_red = cv2.inRange(image_hsv, low_apple_red, high_apple_red)
    mask_raw = cv2.inRange(image_hsv, low_apple_raw, high_apple_raw)
    mask_green = cv2.inRange(image_hsv, low_apple_green, high_apple_green)

    # Combine the masks to get the final mask.
    mask = mask_red + mask_raw + mask_green

    # Adjust kernel size according to the apple size in the image.
    kernel_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply morphological operations to clean up the mask.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask.
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    c_num = 0
    for i, c in enumerate(cnts):
        # Draw a circle around the detected apple.
        ((x, y), r) = cv2.minEnclosingCircle(c)
        if r > 10:  # Filter out small contours that are not apples.
            c_num += 1
            cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.putText(image, "#{}".format(c_num), (int(x) - 10, int(y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            continue

    # Compare detected apple count with ground truth.
    if c_num == apple_count:
        corret_num += 1

# Calculate the accuracy of apple detection.
correct_rate = 100 * corret_num / total_num

print(correct_rate)

# Uncomment these lines to view the images with detections.
# cv2.imshow("Original image", image_bgr)
# cv2.imshow("Detected Apples", image)
# cv2.imshow("HSV Image", image_hsv)
# cv2.imshow("Mask image", mask)
# cv2.waitKey(0)

# Uncomment these lines to save the images.
# cv2.imwrite("image6.png", image_bgr)
# cv2.imwrite("pic6.png", image)
# cv2.imwrite("hsv.png", image_hsv)
# cv2.imwrite("mask.png", mask)