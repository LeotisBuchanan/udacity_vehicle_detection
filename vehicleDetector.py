import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from featuregenerator import FeatureGenerator, single_img_features
from predictionQualityManager import PredictionQualityManager
from windowManager import WindowManager
from classifier import Classifier
from settings import Settings
import sys
from moviepy.editor import VideoFileClip
import imageio
from sklearn.externals import joblib


class VehicleDetector:
    def __init__(self):
        DATA_BASE_PATH = "data/"
        CARS = "cars1/"
        NOT_CARS = "notcars1/"
        self.cars_base_path = DATA_BASE_PATH + CARS
        self.not_cars_base_path = DATA_BASE_PATH + NOT_CARS
        self.classifier = Classifier()
        self.classifier.train(self.cars_base_path, self.not_cars_base_path,retrain=False)
        self.windowManager = WindowManager()
        self.settingsDict = (Settings()).settingsDict
        self.featureGenerator = FeatureGenerator(self.settingsDict)
        self.predictionQualityManager = PredictionQualityManager()
        
    def pipeline(self, img):
       
        # only search the bottom part of the image
        ## search the bottom part of the with a large window
        # to detect cars that are close
        # search the middle with smaller windows to detect
        # cars that are far away
        # also clip the sides of the image to exclude areas
        # that are off the road
        
        y_stop = int(img.shape[0]*0.1)
        candidate_windows = self.windowManager.slide_window(img,
                                                   x_start_stop=[None, None],
                                                   y_start_stop=[y_stop, None], 
                                                   xy_window=(64, 64),
                                                   xy_overlap=(0.5, 0.5))

        candidate_windows128 = self.windowManager.slide_window(img,
                                                   x_start_stop=[None, None],
                                                   y_start_stop=[y_stop, None], 
                                                   xy_window=(128, 128),
                                                   xy_overlap=(0.5, 0.5))

        candidate_windows.extend(candidate_windows128)

        # return a list of boxes coordinates
        detected_cars_coordinates = self.windowManager.search_windows(img,
                                                                 candidate_windows,
                                                                 self.classifier,
                                                                 self.featureGenerator,
                                                                 self.settingsDict)


        best_pred_bboxes = self.predictionQualityManager.findBestPredictions(
            detected_cars_coordinates)

        print(candidate_windows)
        output_img = self.windowManager.draw_boxes(img, best_pred_bboxes,
                                          color=(0, 0, 255), thick=6)
    
        return output_img



    def run(self, videoList):
        for video in videoList:
            s = video.split("/")
            s[1] = "output"
            print("video:" , video)
            print("s:" , s)
            output =  "/".join(s)
            print("output:", output)
            clip2 = VideoFileClip(video)
            clip = clip2.fl_image(self.pipeline)
            clip.write_videofile(output, audio=False)    

if __name__ == "__main__":
    vehicleDetector = VehicleDetector()
    #videoList = ["videos/input/test_video.mp4", "videos/input/project_video.mp4"]
    #vehicleDetector.run(videoList)

    img = mpimg.imread('test_images/test2.jpg')
    image = vehicleDetector.pipeline(img)
    print("showing image")
    plt.imshow(image)
    plt.show()
