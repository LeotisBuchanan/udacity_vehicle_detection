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


    def computeBoxes(self, img):
        x_stop = img.shape[1]
        x_start = 100
        windowManager = self.windowManager
        c64 = windowManager.slide_window(img,
                                          x_start_stop=[x_start ,  x_stop],
                                          y_start_stop=[400, 500], 
                                          xy_window=(64, 64),
                                          xy_overlap=(0.5, 0.5))

        c128 = windowManager.slide_window(img,
                                          x_start_stop=[x_start ,  x_stop],
                                          y_start_stop=[300, 600], 
                                          xy_window=(128,128),
                                          xy_overlap=(0.2, 0.2))



        c512 = windowManager.slide_window(img,
                                           x_start_stop=[x_start ,  x_stop],
                                           y_start_stop=[350,730], 
                                           xy_window=(256, 256),
                                           xy_overlap=(0.5, 0.5))

        # cw =  c96 + c64 + c512
        cw =   c64  #+ c128 + c512

        return c128 + c512 + c64

    
        
    def pipeline(self, img, candidate_windows, classifier, featureGenerator, settingsDict):
       
        # return a list of boxes coordinates]
        detected_cars_coordinates = self.windowManager.search_windows(img,
                                                                 candidate_windows,
                                                                 classifier,
                                                                 featureGenerator,
                                                                 settingsDict)

        img  = self.predictionQualityManager.findBestPredictions(img,
            detected_cars_coordinates)

        return img

    def run(self, videoList, classifier, featureGenerator, settingsDict):
        # get a sample image
        img = mpimg.imread('test_images/test1.jpg')
        candidate_windows = vehicleDetector.computeBoxes(img)
        for video in videoList:
            s = video.split("/")
            s[1] = "output"
            output = "/".join(s)
            clip2 = VideoFileClip(video)
            clip = clip2.fl_image(lambda image: self.pipeline(image,
                                                              candidate_windows,
                                                              classifier,
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
        heatmap = None
        heatmaps = []
        for img in testImages:
            image, heatmap = vehicleDetector.pipeline(img, classifier,
                                             featureGenerator, settingsDict)
            resultImages.append((image,heatmap))
        
        """
        # a = fig.add_subplot(1, 3, 1)
        imgplot = plt.imshow(resultImages[0])
        imgplot.figure.savefig("image1.png")
        
        a.set_title("test1")
        a=fig.add_subplot(1,3,2)
        """
        heatplot = plt.imshow(resultImages[5][1])
        heatplot.figure.savefig("heatmap.jpg", cmap='hot')

        imgplot = plt.imshow(resultImages[5][0])
        imgplot.figure.savefig("image.jpg")
        
        """
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
        imgplot.figure.savefig("image6.png")
        """
        
        # plt.ion()
        # plt.show()
        
            
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
