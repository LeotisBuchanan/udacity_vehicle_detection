import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from skimage.feature import hog
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from settings import Settings
from featuregenerator import FeatureGenerator
import glob


##  Todo need to create a pipeline with
##  with a scalar so that I can save the
##  classifier 

class Classifier:

    def __init__(self):
        self.svc = LinearSVC()

    def classify(self, features):
        # prediction = self.svc.predict(features)
        decision_threshhold = self.svc.decision_function(features)
        prediction = int(decision_threshhold > 1)
        return prediction

    def train(self, cars_base_path, not_cars_base_path, retrain=True):
       classifier = None
       if retrain:
           classifier = self._train(cars_base_path, not_cars_base_path)
           # save classifier for later use
           joblib.dump(classifier, 'vehicle_classifier.pkl')
       else:
           # load the existing classifier
           classifier = joblib.load('vehicle_classifier.pkl') 

    
    def _train(self, cars_base_path, not_cars_base_path):

        settings = Settings()
        settingsDict = settings.settingsDict
        featureGenerator = FeatureGenerator(settingsDict)

        cars_image_path_list = glob.glob(cars_base_path+"/*.jpeg")
        not_cars_image_path_list = glob.glob(not_cars_base_path+"/*.jpeg")


        car_features = featureGenerator.computeFeatures(cars_image_path_list)
        not_car_features = featureGenerator.computeFeatures(not_cars_image_path_list)
    
        X = np.vstack((car_features,
                       not_car_features)).astype(np.float64)                        


        # Fit a per-column scaler
        self.scaler = StandardScaler().fit(X)

        # Apply the scaler to X
        scaled_X = self.scaler.transform(X)
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)),
                       np.zeros(len(not_car_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)
        # Check the training time for the SVC
        self.svc.fit(X_train, y_train)
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))    
        return self

