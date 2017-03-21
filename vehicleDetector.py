import matplotlib.image as mpimg
import numpy as np
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def hog():
    # Take histograms in R, G, and B
    rhist = np.histogram(image[:,:,0], bins=32, range=(0, 256))
    ghist = np.histogram(image[:,:,1], bins=32, range=(0, 256))
    bhist = np.histogram(image[:,:,2], bins=32, range=(0, 256))




def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img




    


def run_pipeline():
# Read in the image    
    image = mpimg.imread('test_images/bbox-example-image.jpg')
    bboxes = [((275, 572), (380, 510)), ((488, 563), (549, 518)), ((554, 543), (582, 522)), 
      ((601, 555), (646, 522)), ((657, 545), (685, 517)), ((849, 678), (1135, 512))]
    result = draw_boxes(image, bboxes)
    plt.imshow(result)
    plt.show()

if __name__ == "__main__":
    run_pipeline()
