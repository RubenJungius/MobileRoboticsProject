import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, Button

def tuning_done(root):
    # Close the current window
    root.quit()

############################ red ###############################

def update_parameters_red(red_threshold_slider, image_capture):
    # Get the current values from the sliders
    red_threshold = red_threshold_slider.get()

    # Capture a frame from the webcam
    #_, frame = image_capture.read()
    frame=image_capture

    # Apply the updated parameters and display the updated image
    result_image = process_image_red(frame, red_threshold)
    cv2.imshow("Updated Image", result_image)
    cv2.imwrite("Obstacle_map.png",result_image)

def process_image_red(image, red_threshold):
    red_channel = image[:, :, 0].copy()  # Create a copy of the red channel
    red_channel[red_channel > red_threshold] = 0
    _, binary = cv2.threshold(red_channel, 1, 255, cv2.THRESH_BINARY)
    return binary

def red_binarisation(image_capture):
    # Create the main window
    root = tk.Tk()
    root.title("Parameter Adjustment")

    # Create sliders for parameter adjustment
    red_threshold_slider = Scale(root, from_=0, to=255, label="Red Threshold", orient="horizontal")
    red_threshold_slider.set(110)
    red_threshold_slider.pack()

    update_button = tk.Button(root, text="Update Image", command=lambda: update_parameters_red(red_threshold_slider, image_capture))
    update_button.pack()

    done_button = Button(root, text="Tuning Done", command=lambda: tuning_done(root))
    done_button.pack()

    # Display the initial image
    #_, initial_frame = image_capture.read()
    cv2.imshow("Original Image", image_capture)

    # Start the Tkinter main loop
    root.mainloop()

############################ blue ###############################

def update_parameters_blue(low_threshold_slider, high_threshold_slider, image_capture):
    # Get the current values from the sliders
    low_threshold = low_threshold_slider.get()
    high_threshold = high_threshold_slider.get()

    # Capture a frame from the webcam
    frame = image_capture

    # Apply the updated parameters and display the updated image
    result_image = process_image_blue(frame, low_threshold, high_threshold)
    cv2.imshow("Updated Image", result_image)
    cv2.imwrite("end_point.png",result_image)

def process_image_blue(image, low_threshold, high_threshold):
    blue_channel = image[:, :, 2]
    binary = ((blue_channel > low_threshold) & (blue_channel < high_threshold)).astype(np.uint8) * 255
    return binary

def blue_binarisation(image_capture):
    # Create the main window
    root = tk.Tk()
    root.title("Parameter Adjustment")

    # Create sliders for parameter adjustment
    low_threshold_slider = Scale(root, from_=0, to=255, label="Low Blue Threshold", orient="horizontal")
    low_threshold_slider.set(110)
    low_threshold_slider.pack()

    high_threshold_slider = Scale(root, from_=0, to=255, label="High Blue Threshold", orient="horizontal")
    high_threshold_slider.set(150)
    high_threshold_slider.pack()

    update_button = tk.Button(root, text="Update Image", command=lambda: update_parameters_blue(low_threshold_slider, high_threshold_slider, image_capture))
    update_button.pack()

    done_button = Button(root, text="Tuning Done", command=lambda: tuning_done(root))
    done_button.pack()

    # Display the initial image
    #_, initial_frame = image_capture.read()
    cv2.imshow("Original Image", image_capture)

    # Start the Tkinter main loop
    root.mainloop()