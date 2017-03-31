import os
import glob
import time
import itertools
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix
import numpy as np
from featuregenerator import FeatureGenerator
from settings  import Settings


class ModelTrainer:

    def train(self, cars_base_path, not_cars_base_path):

        classifier = Pipeline([('scaling', StandardScaler()),
                ('classification', LinearSVC(loss='hinge')),
               ])

        
        settings = Settings()
        settingsDict = settings.settingsDict
        featureGenerator = FeatureGenerator(settingsDict)

        cars_image_path_list = glob.glob(cars_base_path+"/*.jpeg")
        not_cars_image_path_list = glob.glob(not_cars_base_path+"/*.jpeg")
        print(len(cars_image_path_list))

        car_features = featureGenerator.computeFeatures(cars_image_path_list)
        not_car_features = featureGenerator.computeFeatures(not_cars_image_path_list)
    
        X = np.vstack((car_features,
                       not_car_features)).astype(np.float64)                        




        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)),
                       np.zeros(len(not_car_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(1000)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)
        # Check the training time for the SVC
        classifier.fit(X_train, y_train)

        
        print('Test Accuracy of SVC = ', round(classifier.score(X_test, y_test), 4))    

        # print confusion matrxi

        y_pred = classifier.predict(X)
        conf = confusion_matrix(y, y_pred)
        print(conf)

        # save the model for later use

        joblib.dump({'model':classifier, 'settings_dict':settingsDict},
                    'models/classifier.pkl')

        return classifier


if __name__ == "__main__":

    DATA_BASE_PATH = "data/"
    CARS = "cars/"
    NOT_CARS = "notcars/"
    cars_base_path = DATA_BASE_PATH + CARS
    not_cars_base_path = DATA_BASE_PATH + NOT_CARS
    modelTrainer = ModelTrainer()
    classifier = modelTrainer.train(cars_base_path, not_cars_base_path)
    
