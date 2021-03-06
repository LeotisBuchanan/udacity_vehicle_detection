import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
from nonmax import non_max_suppression_fast
class PredictionQualityManager:

    def __init__(self, image):
        self.image = image
        self.heat = np.zeros_like(self.image[:,:,0]).astype(np.float)
        self.frame_count  = 0
        
    
    
        
    def findBestPredictions(self, img,detected_cars_bboxes):

        # start counting again
        self.frame_count = self.frame_count + 1
        print("frame_count :" + str(self.frame_count))
        best_predicted_boxes = [] 
        # Add heat to each box in box list
        self.heat = self.add_heat(self.heat, detected_cars_bboxes)
        
        # Apply threshold to help remove false positives
        # only apply this after 15 frames
        if(self.frame_count > 20):
            self.frame_count = 0 
            self.heat = self.apply_threshold(self.heat, 20)
            # Visualize the heatmap when displaying    
        heatmap = np.clip(self.heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        print("Number of cars found:" + str(labels[1]))
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # add boxes  to list
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
            # reset the heatmap
            self.heat = np.zeros_like(self.image[:,:,0]).astype(np.float)
        # Return the image
        return img


        
    def add_heat(self,heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

            # Return updated heatmap
        return heatmap
    
    def apply_threshold(self,heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self,img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
            # Return the image
        return img








