# Vehicle Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

###Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.
I used the function `get_hog_features()` to extract the HOG features from the training data. The steps I took to settle on my final choice of parameters were as follows:
* Visualized HOG gradients of car and not car images with different `cells_per_block` and `pixels_per_cell` settings
* Tested various `color_space`, `cells_per_block`, and `pixels_per_cell` combinations and measured against "time to train",  'time to prediction", and "test accuracy".
* Looked for params that resulted in highest test acuracy without much sacrifice to prediction/training time

HOG Visualization
[HOG Viz](./output_images/hog_viz.png)

###Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained the Classifier using comobined Color and HOG features. I used the features returned from `extract_features_combined()`. The steps taken to train the classifier were as follows:
* find best color classification params
* find best HOG classification params
* combine the extracted features from HOG and color into one feature vector
* use `StandardScaler().fit()` to normalize the features.
* Create label vectors of 1's for `car` and 0's for `not car`
* shuffle and split the training data into training and test data
* use `LinearSVC()` to train the classifier
* measure the accuracy against test with `svc.score()`
* rinse and repeat these steps on a smaller subset of the data until I find an accuracy I'm happy with, then train it on all the data.

__Interesting aside__
Through experimentation with the color features classifier, I discovered that the spatial bins param was not helping the classifier much, given the HLS color space, and 128 histogram bins. For example I achieved highest test accuracy of 97% accuracy on the color features classifier with spatial_bins=1, cspace='HLS', and hist=128.

_However during actual testing against my pipeline, I found that spatial bins did indeed improve the classifier._

###Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?
The function `slide_window()` performs the sliding window search across the image. I also defined a function called `search_image()` which implements `slide_window()` at different scales.

I started with a basic imlementation of `search_image()` and tested it with my detection pipeline. Then I iterated through my detection pipeline with implementations of `search_image()` that included multiscale windows. I kept small windows near the horizon and the larger windos near the forground. I iterated on the detection pipeline with the test images until I found a combination of scales that appeared to work well. I found that having many windows greatly improved the detection pipeline. It also greatly reduced the speed. Finding the right balance between speed and accuracy was the greatest challenge in this project.

###Show some examples of test images to demonstrate how your pipeline is working. How did you optimize the performance of your classifier?
Here is the basic result from `slide_window()` with no params
[sliding window no params](./output_images/sliding_window_no_params.png)

Here is some vehicle detection after using multiscale sliding windows. The image below shows the pipeline not producing any false positives
[detection with overlap](./output_images/detection_with_overlap.png)

Here is a heatmap projection of overlapping bounding boxes.
[Heatmap](./output_images/heatmap.png)

Here is what the final output of the pipeline looks like after drawing bounding boxes on the labeled image.
[Final Out](./output_images/final_output.png)

###Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
[Project Video](https://youtu.be/dsH-FK5Yog4)

###Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
I used the heatmap method for combining overlapping bounding boxes and removing false positives. To create the heatmap I defined a function `add_heat()` which takes in a zeroed image array and a list of all the detected bounding boxes. The function adds +1 to every pixel in each of the detected bounding boxes. Then I use `apply_threshold()` to zero out the pixels that do not meet the threshold which are presumably the false positives.

I take advantage of scipy's `labels()` function to easily transform the heatmap into a labeled map. This allows me to remove overlapping bounding boxes. I defined a function called `draw_labeled_bboxes()` which takes in the labeled map and draws bounding boxes along the min/max x and y positions of the nonzero pixels in the labeled map.

###Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?
The shadows projected from the tress onto the road proved once again to be a challenge for this project. My first implementations produced many false positives along that section of road. 

My implementation is inefficent at extracting HOG features. To improve it, I could create the gradient just once for the entire region of interest, set the param `feature_vec=False`, and then subsample that array for each sliding window. However, I did not want to invest the time to implement this solution.

My implementation is also could also improve by averaging the bounding boxes over time or
