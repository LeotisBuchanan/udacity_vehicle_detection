import numpy as np
import cv2
from featuregenerator import FeatureGenerator

class WindowManager:

    # Define a function that takes an image,
    # start and stop positions in both x and y, 
    # window size (x and y dimensions),  
    # and overlap fraction (for both x and y)

    def slide_window(self, img, x_start_stop=[None, None],
                     y_start_stop=[None, None], 
                     xy_window=(64, 64),
                     xy_overlap=(0.5, 0.5)):

        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] is None:
            x_start_stop[0] = 0
        if x_start_stop[1] is None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] is None:
            y_start_stop[0] = 0
        if y_start_stop[1] is None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
        # Initialize a list to append window positions to
        window_list = []

        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def search_windows(self, img, windows, classifier,
                       featureGenerator, settingsDict):

        on_windows = []
        # 2) Iterate over all windows in the list
        for window in windows:
            # 3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1],
                                      window[0][0]:window[1][0]], (64, 64))      
            # 4) Extract features for that window using single_img_features()
            features = featureGenerator.getAllFeatures(test_img, settingsDict)
            # 5) Scale extracted features to be fed to classifier
            
            test_features = np.array(features).reshape(1, -1)
            # 6) Predict using your classifier

            decision = classifier.decision_function(test_features)
            prediction = int(decision > 0.1)
            if prediction == 1:
                on_windows.append(window)
        # 8) Return windows for positive detections
        return on_windows


    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
            # Return the image copy with boxes drawn
        return imcopy


