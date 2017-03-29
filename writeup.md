## TODO 
1. INCLUDE VIDEO OF PIPELINE OUTPUT
2. HISTOGRAM OF HOG
3. SLIDE WINDOW SEARCH
4. INCLUDE IMAGES OF PIPELINE WORKING


**Vehicle Detection Project**

### Histogram of Oriented Gradients (HOG)

Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.

response
Explanation given for methods used to extract HOG features, including which color space was chosen, which HOG parameters (orientations, pixels_per_cell, cells_per_block), and why.


---
###Writeup 


Sliding Window Search

Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?


A sliding window approach has been implemented, where overlapping tiles in each test image are classified as vehicle or non-vehicle. Some justification has been given for the particular implementation chosen.


Show some examples of test images to demonstrate how your pipeline is working. How did you optimize the performance of your classifier?

Some discussion is given around how you improved the reliability of the classifier i.e., fewer false positives and more reliable car detections (this could be things like choice of feature vector, thresholding the decision function, hard negative mining etc.)


## Video Implementation

Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

The following a youtube link to the video output of my pipeline
<link to youtube video>


The sliding-window search plus classifier has been used to search for and identify vehicles in the videos provided. Video output has been generated with detected vehicle positions drawn (bounding boxes, circles, cubes, etc.) on each frame of video.


Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used the approached provided in the lectures.The approach basically assumes that an object that is classified as a car frequently can be assumed to be a car. I think this reasoning is flawed. 

But I used it anyway, time did not permit me to implement a better approach. I do however think that there are more robust approaches. The code that does this is abstracted 
nicely in the predictionQualityManager.py file in the PredictionQualityManager class.

The method used was adopted from the lecture notes and is as following:
- only search the areas where cars are expected to be or where it matters i.e is on the road and it the vehicle path
- since cars do not just appear and then disappear, count the number of times and object is detected and classified 
  as a car. If it repeatedly classified as a car then it just might be car. 
- since cars can be very near, far or very far, search the frame using  sub windows of different sizes
- for detection windows that overlap, find the centroid of the windows and use that as the location of the detected car.

<show images of the output of the approach>


Discussion

Briefly discuss any problems / issues you faced in your implementation of this project. 

Problems: 
- Throughout this term, my major problem was getting time to work on these projects, this severly affected the quality of 
  my submissions. 
- Most of the code was provided by the instructor, this posed a problem for me, because once i read code, it affected 
  my ability to come up with an independent solution. 


Where will your pipeline likely fail? 
I dont think the approach taken in this project is very robust. Our approach could be made to fail by a picture 
of a car on a wall or the back of a truck. Also it is dependent on vehicles have markings/edges that are different 
from other objects in the camera view. This is not very robust or future prove, because if a vehicle looks like non-cars 
then it will not be classified as a car. We did not address ambient lighting, like glare from head lamps, or no lighting, 
cars without headlamps driving in the night(yes it does happen :-))


What could you do to make it more robust?
The approach taken in this project, does not allow the system to really understand what a vehicle is. Incoperating other features of a vehicle such:
- its on the road, 
- its behavior it is moving
- it structure i.e it has wheels
- other characteristics that humans use to determine that a vehicle is a vehicle.


The algorithm used assumes laboratory like conditions, i.e
- the lighting is at a particular level
- the images possess particular features
- it did not take in account other types of vehicles, trucks, bicycles, motor cycles

These issues may cause the pipeline to fail, when they occur in the field. for example a car painted with trees with its head lights off. This would probably
not be recognized as car. 

Additionally the pipeline needs to faster, this can be achieved by parallizing the pipeline, or by rewriting it in C or assembly (:-))


The pipeline can be improved by making it a part of system:
- that takes into account the  features of  global environment in which the car is operating.
- that learns and make inferences about the environment
- that understands the concepts of vehicle, driving, road vs not road


