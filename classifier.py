import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from skimage.feature import hog
import cv2
from sklearn.preprocessing import StandardScaler


class Classifier:

    def __init__(self):
        self.svc = LinearSVC()

    def classify(self, features):
        prediction = self.svc.predict(features)
        return prediction


    def train(self, car_features, not_car_features):
        import sys

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
        
