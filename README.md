## Plankton image classification

This repo contains scripts for the Kaggle National Data Science Bowl 
competition ([1]), where the objective was to develop an algorithm to 
predict the probability that a given image belongs to one of 
121 different classes, most of which are varieties of plankton.
Multiclass log loss was used for evaluation.

### Files

- `tutorial.ipynb`: duplicates the tutorial code provided for the competition
                    and extends it to predict probabilities for the test data
- `image_files.py`: defines classes for working with individual images
  		    and the training and test collections of images
- `feature_table.py`: defines a class that computes features and organizes
  		      them in a data frame
- `get_features.py`: defines functions for loading precomputed features or
  		     computing new ones for the training and test data
- `explore.py`: includes various functions for exploratory analysis of 
  		images and features
- `feature_mean_var_by_class.py`: a short script to compute a statistic to
  				  rank features
- `logr.py`, `knn.py`, `rf.py`, `svm.py`: scripts for training various models,
  	     	       			  optimizing their parameters using
					  cross-validation, and computing
					  predicted class probabilites for the
					  test images
- `learning_curve.py`: plots learning curves using multiclass log loss

### Evaluation

The baseline (uniform probability for all classes) results in a multiclass
log loss of about 4.8. The tutorial, which uses 25x25 pixels to represent
each image and adds a feature based on the ratio of major and minor axes
of the largest region, reduces the log loss to about 3.7. The best models
from the scripts included here achieve a log loss around 2.0 using 
a random forest or SVM with a radial basis function and a set of
10 features computed from the largest region in each image.


[1]: https://www.kaggle.com/c/datasciencebowl
