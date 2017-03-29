# rass
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

class VehicleDetector:
    def __init__(self):
        DATA_BASE_PATH = "data/"
        CARS = "cars1/"
        NOT_CARS = "notcars1/"

        self.cars_base_path = DATA_BASE_PATH + CARS
        self.not_cars_base_path = DATA_BASE_PATH + NOT_CARS

        self.test_image = mpimg.imread('data/bbox-example-image.jpg')

            
    def getImagesPath(self,image_directory):
        image_paths = glob.glob(image_directory+"/*.jpeg")
        return image_paths


    def loadData(self,base_path):
        image_paths = self.getImagesPath(base_path)
        return image_paths


    def pipeline(self,img):

        # step 1 load the data
        # step 2 compute features
        # step 3 train classifier
        # steo 4 get windows
        # step 3 prediction
        # step 4 visualise predictions

    
        settings = Settings()
        settingsDict = settings.settingsDict
        featureGenerator = FeatureGenerator(settingsDict)
        windowManager = WindowManager()
        classifier = Classifier()
        predictionQualityManager = PredictionQualityManager()
        not_cars = self.loadData(self.not_cars_base_path)
        cars = self.loadData(self.cars_base_path)
        car_features = featureGenerator.computeFeatures(cars)
        non_car_features = featureGenerator.computeFeatures(not_cars)
        # need to reshape the feature list to be a single
        # column array
        classifier.train(car_features, non_car_features)

        # only search the bottom part of the image
        y_stop = int(img.shape[0]*0.7)
        candidate_windows = windowManager.slide_window(img,
                                                   x_start_stop=[None, None],
                                                   y_start_stop=[y_stop, None], 
                                                   xy_window=(64, 64),
                                                   xy_overlap=(0.5, 0.5))




        candidate_windows128 = windowManager.slide_window(test_image,
                                                   x_start_stop=[None, None],
                                                   y_start_stop=[y_stop, None], 
                                                   xy_window=(128, 128),
                                                   xy_overlap=(0.5, 0.5))

    
        candidate_windows.extend(candidate_windows128)

        # return a list of boxes coordinates
        detected_cars_coordinates = windowManager.search_windows(test_image,
                                                             candidate_windows,
                                                             classifier,
                                                             featureGenerator,
                                                             settingsDict)


        best_pred_bboxes = predictionQualityManager.findBestPredictions(
            detected_cars_coordinates)
        output_img = windowManager.draw_boxes(test_image, best_pred_bboxes,
                                          color=(0, 0, 255), thick=6)
    
        plt.imshow(output_img)
        plt.show()




    def run(self, videoList):
        for video in videoList:
            output =  "output" + video
            clip2 = VideoFileClip(video)
            clip = clip2.fl_image(self.pipeline)
            clip.write_videofile(output, audio=False)    

if __name__ == "__main__":
    vehicleDetector = VehicleDetector()
    videoList = ["videos/input/test_video.mp4", "videos/input/project_video.mp4"]
    vehicleDetector.run(videoList)
    


