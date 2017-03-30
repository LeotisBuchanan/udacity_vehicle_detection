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
        self.classifier.train(self.cars_base_path, self.not_cars_base_path)
        self.windowManager = WindowManager()
        self.settingsDict = (Settings()).settingsDict
        self.featureGenerator = FeatureGenerator(self.settingsDict)
        template_image = mpimg.imread('test_images/test5.jpg')        
        self.predictionQualityManager = PredictionQualityManager(template_image)
        
    def pipeline(self, img):
       
        # only search the bottom part of the image
        ## search the bottom part of the with a large window
        # to detect cars that are close
        # search the middle with smaller windows to detect
        # cars that are far away
        # also clip the sides of the image to exclude areas
        # that are off the road
        
        y_stop = int(img.shape[0]*0.1)
        y_stop = int(img.shape[0]*0.4)
        print("y_stop:", y_stop)
        
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


    def testOnImages(self):
        img1 = mpimg.imread('test_images/test1.jpg')
        img2 = mpimg.imread('test_images/test2.jpg')
        img3 = mpimg.imread('test_images/test3.jpg')
        img4 = mpimg.imread('test_images/test4.jpg')
        img5 = mpimg.imread('test_images/test5.jpg')
        img6 = mpimg.imread('test_images/test5.jpg')

        testImages = [
            img1,
            img2,
            img3,
            img4,
            img5,
            img6
        ]

        resultImages = []
        
        for img in testImages:
            image = vehicleDetector.pipeline(img)
            resultImages.append(image)
        

        fig = plt.figure()
        a=fig.add_subplot(1,3,1)
        imgplot = plt.imshow(resultImages[0])
        a.set_title("test1")
        a=fig.add_subplot(1,3,2)
        imgplot = plt.imshow(resultImages[1])

        a=fig.add_subplot(1,3,3)
        imgplot = plt.imshow(resultImages[2])
        a.set_title("test2")
    
        a=fig.add_subplot(2,3,1)
        imgplot = plt.imshow(resultImages[3])
        a.set_title("test3")
    
        a=fig.add_subplot(2,3,2)
        imgplot = plt.imshow(resultImages[4])
        a.set_title("test4")
    
        a=fig.add_subplot(2,3,3)
        imgplot = plt.imshow(resultImages[5])
        a.set_title("test5")

    
        print("showing image")
        plt.show()
        
            
if __name__ == "__main__":
    vehicleDetector = VehicleDetector()
    videoList = ["videos/input/project_video.mp4"]
    vehicleDetector.run(videoList)

