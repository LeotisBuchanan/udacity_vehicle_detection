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
    print(settingsDict)
    
    featureGenerator = FeatureGenerator(settingsDict)

    windowManager = WindowManager()
    classifier = Classifier()
    predictionQualityManager = PredictionQualityManager()

    not_cars = loadData(not_cars_base_path)
    cars = loadData(cars_base_path)
    cars = cars[0:1]
    print("### my generator ####")
    car_features = featureGenerator.computeFeatures(cars)
    print(car_features.shape)

    print("udacity generated features")
    img = mpimg.imread(cars[0])
    f = single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                            hist_bins=32, orient=9, 
                            pix_per_cell=8, cell_per_block=2, hog_channel=0,
                            spatial_feat=True, hist_feat=True, hog_feat=True)
    
    print(f.shape)



    sys.exit(0)
    non_car_features = featureGenerator.computeFeatures(not_cars)

    classifier.train(car_features, non_car_features)

    candidate_windows = windowManager.slide_window(test_image,
                                                   x_start_stop=[None, None],
                                                   y_start_stop=[None, None], 
                                                   xy_window=(64, 64),
                                                   xy_overlap=(0.5, 0.5))

    # return a list of boxes coordinates
    detected_cars_coordinates = windowManager.search_windows(test_image,
                                                             candidate_windows,
                                                             classifier)
    best_pred_bboxes = predictionQualityManager.findBestPredictions(
        detected_cars_coordinates)
    output_img = windowManager.draw_boxes(test_image, best_pred_bboxes,
                                          color=(0, 0, 255), thick=6)
    
    plt.imshow(output_img)
    plt.show()


if __name__ == "__main__":
    runpipeline()
    


