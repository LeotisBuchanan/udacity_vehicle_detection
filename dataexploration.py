import glob
from builtins import print
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
import cv2
# images are divided up into vehicles and non-vehicles

DATA_BASE_PATH = "data/"
CARS = "cars1/"
NOT_CARS = "notcars1/"

car_images = glob.glob(DATA_BASE_PATH+CARS + '*.jpeg')
not_car_images = glob.glob(DATA_BASE_PATH+NOT_CARS + '*.jpeg')

cars = []
notcars = []

for c, nc in zip(car_images, not_car_images):
    notcars.append(nc)
    cars.append(c)


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

    
# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict
    

data_info = data_look(cars, notcars)

print('Your function returned a count of', 
      data_info["n_cars"], ' cars and', 
      data_info["n_notcars"], ' non-cars')
print('of size: ', data_info["image_shape"], ' and data type:', 
      data_info["data_type"])
# Just for fun choose random car / not-car indices and plot example images   
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))
    
# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])


# Generate a random index to look at a car image
ind = np.random.randint(0, len(cars))
# Read in the image
image = mpimg.imread(cars[ind])
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Define HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2
# Call our function with vis=True to see an image output
features, hog_image = get_hog_features(gray, orient, 
                                       pix_per_cell, cell_per_block, 
                                       vis=True, feature_vec=False)


# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Visualization')
plt.show()





























