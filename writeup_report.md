## Project: Search and Sample Return
### Writeup Report

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./img_for_report/rock_warped.png
[image2]: ./img_for_report/rock_color_filter.png
[image3]: ./img_for_report/plot_dir.png 


## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
In  `perception_step()`, I created three functions  `color_thresh()`,  `color_thresh_obstacles()`,  `color_thresh_rocks()` to detect navigation, obstavles, rocks respectively. In `color_thresh()`,  `color_thresh_obstacles()`, I use RGB filter as described in the sample. In `color_thresh_rocks()`, I use HSV to get a more robut filter to detect yellow rocks. I also implemented Morphology Transformations to filter some noise. It works better for path finding and obstales detection, but for rocks detection, it doesn't matter so much.

Here is an example of how to include an image in your writeup.

![alt text][image1]
![alt text][image2]
![alt text][image3]

#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

The `process_image()` here is similar to what we would use in autonomous navigation model.
##### 1) Define source and destination points for perspective transform
This is the calibrtion step to get matrix M.
##### 2) Apply perspective transform
##### 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
##### 4) Convert thresholded image pixel values to rover-centric coords
##### 5) Convert rover-centric pixel values to world coords
Here I put navigation, obstacle, rocks object in color channel 2,0,1.
##### 6) Update worldmap (to be displayed on right side of screen)
##### 7) Make a mosaic image
Make a image to include original image, world map image and rover-centric image.

The output video `my_mapping.mp4` for my dataset in `my_dataset` can be found in  `../output`.

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.
I didn't change so much from `process_image()` in Jupyter notebook, but I move the calibration part out of `perception_step()` and calcuate it in the beginning of `driver_rover()` to save the matrix M in `Rover.M` for efficiency. 
I also modified decision making tree, introducing `Rover.stop_front_thresh` to measure the front distance from the rover to wall to give a better feedback of anti-collision. 

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

I use 1024x768 resolution and fastest graphic quality option. FPS is 18.
 
The rover achieved at least 70% of the environment with 65% fidelity (accuracy) against the ground truth and is able to find the location of at least one rock sample. Sometimes the rover can't find a road with a narrow entrance, and collision could still happen in the middle area when there are some small rock obstacles. To solve these issues, next step I will tune the parameters to make a better throttle and brake mode, and advanced opencv method in stead of  `to_polar_coords()` method we use would make the small rock obstacles easier to detect and avoid. For the road with a narrow entrance, search algorithm would fix that, which will be left for next step.


![alt text][image3]


