Deep Learning for Skin Cancer Detection.

This file explains all classification codes, including their requirements and running procedure.


Training Instructions:

Step 1. "python parsing_train_data.py"

  Requires training images and Ground truth labels to be stored at specific location (see code) as in format provided by ISBI Challenge 2016.
  
  Stores Parsed training set and ground truths in .txt format.
  
Step 2. "python vgg_training.py" for VGG-16 model or "python vgg19_training.py" for VGG-19 model.



Testing Instructions:

Step 1. "python parsing_test_data.py"

Step 2. "python vgg_test_final.py" calculates probability for each file in test dataset

Step 3. "vgg_submission.py" script to finally convert results to submission format for ISBI Challenge 2016.
