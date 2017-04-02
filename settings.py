

class Settings:
    def __init__(self):
        self.settingsDict = {}
        constants = Constants()
        self.settingsDict[constants.COLOR_SPACE] = 'YCrCb'
        self.settingsDict[constants.SPATIAL_SIZE] = (32, 32)
        self.settingsDict[constants.HIST_BINS] = 16
        self.settingsDict[constants.ORIENTATION] = 9
        self.settingsDict[constants.PIXEL_PER_CELL] = 8
        self.settingsDict[constants.CELL_PER_BLOCK] = 2
        self.settingsDict[constants.HOG_CHANNEL] = "ALL"
        self.settingsDict[constants.SPATIAL_FEATURE] = True
        self.settingsDict[constants.HIST_FEATURE] = True
        self.settingsDict[constants.HOG_FEATURES] = True
        self.settingsDict[constants.Y_START_STOP] = [450, 680]
        self.settingsDict[constants.COLOR_HIST_BINS] = 32
        self.settingsDict[constants.COLOR_HIST_BINS_RANGE] = (0, 256)
        

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
        self.COLOR_HIST_BINS_RANGE = 10
        self.Y_START_STOP = 11
        self.COLOR_HIST_BINS = 12

