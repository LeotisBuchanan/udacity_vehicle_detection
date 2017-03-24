import glob
from builtins import print
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from skimage.feature import hog
import cv2
from sklearn.preprocessing import StandardScaler


def getImagesPath(image_directory):
    image_paths = glob.glob(image_directory+"/*.jpeg")
    return image_paths


def predict(svc, features):
    prediction = svc.predict([features])
    return prediction


def trainClassifier(X_train, y_train, X_test, y_test):
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    accuracy = round(svc.score(X_test, y_test), 4)
    print("*** I am here ****")
    return svc, accuracy


def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features


# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0],
                                    channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features(imgs, cspace='RGB', orient=9, 
                     pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)      

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(
                    feature_image[:, :, channel], 
                    orient, pix_per_cell, cell_per_block, 
                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(
                feature_image[:, :, hog_channel], orient, 
                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features


def get_hog_features(img, orient, pix_per_cell,
                     cell_per_block, vis=False, feature_vec=True):
    if vis is True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block,
                                                   cell_per_block),
                                  transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell,
                                        pix_per_cell),
                       cells_per_block=(cell_per_block,
                                        cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features


def runpipeline():
    # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    colorspace = 'RGB'
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # Can be 0, 1, 2, or "ALL"
    hog_channel = 0 

    DATA_BASE_PATH = "data/"
    CARS = "cars1/"
    NOT_CARS = "notcars1/"

    cars_directory = DATA_BASE_PATH + CARS
    non_cars_directory = DATA_BASE_PATH + NOT_CARS

    # get cars path
    cars = getImagesPath(cars_directory)
    not_cars = getImagesPath(non_cars_directory)

    # Reduce the sample size because HOG features are slow to compute
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 500
    cars = cars[0:sample_size]
    not_cars = not_cars[0:sample_size]

    # generate features
    car_features = extract_features(cars, cspace=colorspace, orient=orient, 
                                    pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel)
    not_car_features = extract_features(not_cars, cspace=colorspace,
                                        orient=orient, 
                                        pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block, 
                                        hog_channel=hog_channel)

    X = np.vstack((car_features, not_car_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)),
                   np.zeros(len(not_car_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    model, accuracy = trainClassifier(X_train, y_train, X_test, y_test)
    print(accuracy)
    print(model)


if __name__ == "__main__":
    runpipeline()



