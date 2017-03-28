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


def getImagesPath(image_directory):
    image_paths = glob.glob(image_directory+"/*.jpeg")
    return image_paths


def loadData(base_path):
    image_paths = getImagesPath(base_path)
    return image_paths


def runpipeline():

    # step 1 load the data
    # step 2 compute features
    # step 3 train classifier
    # steo 4 get windows
    # step 3 prediction
    # step 4 visualise predictions

    DATA_BASE_PATH = "data/"
    CARS = "cars1/"
    NOT_CARS = "notcars1/"

    cars_base_path = DATA_BASE_PATH + CARS
    not_cars_base_path = DATA_BASE_PATH + NOT_CARS

    test_image = mpimg.imread('data/bbox-example-image.jpg')
    img = test_image
    
    settings = Settings()
    settingsDict = settings.settingsDict
    featureGenerator = FeatureGenerator(settingsDict)
    windowManager = WindowManager()
    classifier = Classifier()
    predictionQualityManager = PredictionQualityManager()
    not_cars = loadData(not_cars_base_path)
    cars = loadData(cars_base_path)
    car_features = featureGenerator.computeFeatures(cars)
    non_car_features = featureGenerator.computeFeatures(not_cars)
    # need to reshape the feature list to be a single
    # column array
    classifier.train(car_features, non_car_features)

    candidate_windows = windowManager.slide_window(test_image,
                                                   x_start_stop=[None, None],
                                                   y_start_stop=[None, None], 
                                                   xy_window=(64, 64),
                                                   xy_overlap=(0.5, 0.5))


    

    # return a list of boxes coordinates
    detected_cars_coordinates = windowManager.search_windows(test_image,
                                                             candidate_windows,
                                                             classifier,
                                                             featureGenerator,
                                                             settingsDict)

    sys.exit(0)
    best_pred_bboxes = predictionQualityManager.findBestPredictions(
        detected_cars_coordinates)
    output_img = windowManager.draw_boxes(test_image, best_pred_bboxes,
                                          color=(0, 0, 255), thick=6)
    
    plt.imshow(output_img)
    plt.show()


if __name__ == "__main__":
    runpipeline()
    


