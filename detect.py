import cv2
import csv

data = {}
file_path = 'ground_truth.csv'

# Defining the color ranges to be filtered.
# The following ranges should be used on HSV domain image.
low_apple_red = (160.0, 153.0, 153.0)
high_apple_red = (180.0, 255.0, 255.0)
low_apple_raw = (0.0, 150.0, 150.0)
high_apple_raw = (15.0, 255.0, 255.0)
low_apple_green = (25.0, 100.0, 100.0)
high_apple_green = (70.0, 255.0, 255.0)

# Open CSV file
with open(file_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)

    # Iterate through each row of a CSV file
    for row in csv_reader:
        # The first column is the image file name and the second column is the number of apples
        image_name = row[0]
        apple_count = int(row[1])

        # Store data into dictionary
        data[image_name] = apple_count

corret_num = 0
total_num = len(data)

for image_name, apple_count in data.items():

    image_path = 'counting/images/' + image_name

    image_bgr = cv2.imread(image_path)
    image = image_bgr.copy()
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    mask_red = cv2.inRange(image_hsv, low_apple_red, high_apple_red)
    mask_raw = cv2.inRange(image_hsv, low_apple_raw, high_apple_raw)
    mask_green = cv2.inRange(image_hsv, low_apple_green, high_apple_green)

    mask = mask_red + mask_raw + mask_green

    # The size of the core should be adjusted according to the size of the apple
    kernel_size = 5 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    c_num=0
    for i,c in enumerate(cnts):
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        if r>10:
            c_num+=1
            cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.putText(image, "#{}".format(c_num), (int(x) - 10, int(y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            continue
    if(c_num == apple_count):
        corret_num += 1

correct_rate = 100*corret_num/total_num

print(correct_rate)

# cv2.imshow("Original image", image_bgr)
# cv2.imshow("Detected Apples", image)
# cv2.imshow("HSV Image", image_hsv)
# cv2.imshow("Mask image", mask)
# cv2.waitKey(0)
# cv2.imwrite("image6.png", image_bgr)
# cv2.imwrite("pic6.png", image)
# cv2.imwrite("hsv.png", image_hsv)
# cv2.imwrite("mask.png", mask)
