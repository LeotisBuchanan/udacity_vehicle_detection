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
        self.windowManager = WindowManager()
        self.settingsDict = (Settings()).settingsDict
        self.featureGenerator = FeatureGenerator(self.settingsDict)
        template_image = mpimg.imread('test_images/test5.jpg')        
        self.predictionQualityManager = PredictionQualityManager(
            template_image)
        
    def pipeline(self, img, classifier, featureGenerator, settingsDict):
       
        # only search the bottom part of the image
        ## search the bottom part of the with a large window
        # to detect cars that are close
        # search the middle with smaller windows to detect
        # cars that are far away
        # also clip the sides of the image to exclude areas
        # that are off the road

        x_stop = img.shape[1]
        x_start = 0
        candidate_windows64 = self.windowManager.slide_window(img,
                                                              x_start_stop=[x_start ,  x_stop],
                                                              y_start_stop=[400, 500], 
                                                              xy_window=(64, 64),
                                                              xy_overlap=(0.5, 0.5))

        candidate_windows96 = self.windowManager.slide_window(img,
                                                   x_start_stop=[x_start ,  x_stop],
                                                   y_start_stop=[300, 700], 
                                                   xy_window=(150,100),
                                                   xy_overlap=(0.5, 0.5))



        candidate_windows512 = self.windowManager.slide_window(img,
                                                               x_start_stop=[x_start ,  x_stop],
                                                               y_start_stop=[350,730], 
                                                               xy_window=(200, 150),
                                                               xy_overlap=(0, 0))

        #candidate_windows =  candidate_windows96 + candidate_windows512 + candidate_windows64
        candidate_windows =   candidate_windows64
        # return a list of boxes coordinates]
        
        detected_cars_coordinates = self.windowManager.search_windows(img,
                                                                 candidate_windows,
                                                                 classifier,
                                                                 featureGenerator,
                                                                 settingsDict)


        best_pred_bboxes = self.predictionQualityManager.findBestPredictions(
            detected_cars_coordinates)
        

        output_img = self.windowManager.draw_boxes(img,detected_cars_coordinates,
                                                   color=(0, 0, 255), thick=4)
        """
        output_img = self.windowManager.draw_boxes(img, candidate_windows,
                                                   color=(0, 0, 255), thick=4)
        """

        return output_img

    def run(self, videoList, classifier, featureGenerator, settingsDict):
        for video in videoList:
            s = video.split("/")
            s[1] = "output"
            output = "/".join(s)
            clip2 = VideoFileClip(video)
            clip = clip2.fl_image(lambda image: self.pipeline(image, classifier,
                                                         featureGenerator,
                                                         settingsDict))
            clip.write_videofile(output, audio=False)    


    def testOnImages(self,classifier,featureGenerator,settingsDict):
        img1 = mpimg.imread('test_images/test1.jpg')
        img2 = mpimg.imread('test_images/test2.jpg')
        img3 = mpimg.imread('test_images/test3.jpg')
        img4 = mpimg.imread('test_images/test4.jpg')
        img5 = mpimg.imread('test_images/test5.jpg')
        img6 = mpimg.imread('test_images/test5.jpg')

        testImages = [img1,img2,img3,img4,img5,img6]

        resultImages = []
        
        for img in testImages:
            image = vehicleDetector.pipeline(img, classifier,
                                             featureGenerator, settingsDict)
            resultImages.append(image)
        
        fig = plt.figure()

        a = fig.add_subplot(1, 3, 1)
        imgplot = plt.imshow(resultImages[0])
        imgplot.figure.savefig("image1.png")

        a.set_title("test1")
        a=fig.add_subplot(1,3,2)
        imgplot = plt.imshow(resultImages[1])
        imgplot.figure.savefig("image2.png")
        
        a=fig.add_subplot(1,3,3)
        imgplot = plt.imshow(resultImages[2])
        a.set_title("test2")
        imgplot.figure.savefig("image3.png")
    
        a=fig.add_subplot(2,3,1)
        imgplot = plt.imshow(resultImages[3])
        a.set_title("test3")
        imgplot.figure.savefig("image4.png")
    
        a=fig.add_subplot(2,3,2)
        imgplot = plt.imshow(resultImages[4])
        a.set_title("test4")
        imgplot.figure.savefig("image5.png")
    
        a=fig.add_subplot(2,3,3)
        imgplot = plt.imshow(resultImages[5])
        a.set_title("test5")
        imgplot.figure.savefig("image6.png")
        
        plt.ion()
        plt.show()
        
            
if __name__ == "__main__":
    vehicleDetector = VehicleDetector()
    videoList = ["videos/input/project_video.mp4"]
    #videoList = ["videos/input/test_video.mp4"]
    # load the classifier
    classifier_file = "models/classifier.pkl"
    print("loading saved model from file....")
    model_data = joblib.load(classifier_file)
    classifier = model_data['model']
    settingsDict = model_data['settings_dict']                              
    featureGenerator = FeatureGenerator(settingsDict)
    vehicleDetector.run(videoList,classifier, featureGenerator,settingsDict)

    #vehicleDetector.testOnImages(classifier,featureGenerator,settingsDict)
