import cv2
import numpy as np
from skimage.feature import hog


class Constants:
    def __init__(self):
        self.COLOR_SPACE = 0
        self.SPATIAL_SIZE = 1
        self.HIST_BINS = 2
        self.ORIENTATION = 3
        self.PIXEL_PER_CELL = 4
        self.CELL_PER_BLOCK = 5
        self.HOG_CHANNEL = 6
        self.SPATIAL_FEATURE = 7
        self.HIST_FEATURE = 8
        self.HOG_FEATURES = 9
        self.SIZE = 10
        self.BINS_RANGE = 11
        self.NUMBER_OF_BINS = 12
        self.SPATIAL_SIZE = 13


class FeatureGenerator:

    def __init__(self):
        pass

    def getAllFeatures(self, img, settingsDict):

        orient = settingsDict[Constants.ORIENTATION]
        npix_per_cell = settingsDict[Constants.PIXEL_PER_CELL]
        cell_per_block = settingsDict[Constants.CELL_PER_BLOCK]
        size = settingsDict[Constants.SIZE]
        nbins = settingsDict[Constants.NUMBER_OF_BINS]
        bins_range = settingsDict[Constants.BINS_RANGE]

        hgFeatures = self.generate_hog_features(img, orient,
                                                pix_per_cell=pix_per_cell,
                                                cell_per_block=cell_per_block,
                                                vis=False,
                                                feature_vec=True)
        spFeatures = self.generate_spatial_features(img, size=size)
        histFeatures = self.generate_color_histogram_features(
            img,
            nbins=nbins,
            bins_range=bins_range)

        allFeatures = np.concantenate(hgFeatures, spFeatures, histFeatures)

        return allFeatures
        
    def generate_hog_features(self, feature_image, orient,
                              pix_per_cell,hog_channel=0,
                              cell_per_block, vis=False,
                              feature_vec=True):

        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                img_channel = feature_image[:, :, channel]
                features = hog(img_channel, orientations=orient,
                               pixels_per_cell=(pix_per_cell, pix_per_cell),
                               cells_per_block=(cell_per_block, cell_per_block),
                               transform_sqrt=True, 
                               visualise=vis, feature_vector=feature_vec)
                hog_features.extend(features)
                
        else:
            img_channel = feature_image[:, :, hog_channel]
            hog_features = hog(img_channel, orientations=orient,
                               pixels_per_cell=(pix_per_cell, pix_per_cell),
                               cells_per_block=(cell_per_block, cell_per_block),
                               transform_sqrt=True, 
                               visualise=vis, feature_vector=feature_vec)

        return hog_features



    def generate_spatial_features(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel() 
        # Return the feature vector
        return features

    def generate_color_histogram_features(self, img, nbins=32,
                                          bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins,
                                     range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins,
                                     range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins,
                                     range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0],
                                        channel2_hist[0],
                                        channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def convert_color(self, image, colorspace):

        feature_image = np.copy(image)
        if colorspace != 'RGB':
            if colorspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif colorspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif colorspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif colorspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif colorspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        return feature_image
