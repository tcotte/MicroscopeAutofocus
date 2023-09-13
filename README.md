# MicroscopeAutofocus

## Introduction

The aim of this project is to determine the distance on Z axis between current position and focus position using 
single-shot picture. This project is highly inspired by the article **Deep learning-based single-shot autofocus method 
for digital microscopy [1]**. In fact, after we understood that the focus searching through Z stack will be pretty long. We
had to search on an interval of 500µm by a step of 5 to 10µm to retrieve the best Z value to autofocus. This process 
spent around 3 minutes for only one autofocus.

As the article suggests, we first used Mobilnetv3 network to predict the Z autofocus value. Because Mobilnetv3 is a 
classification model, we transformed this model to a regression model changing the last layers. Therefore, the last 
layer contains only one output.

### Dataset

Thanks to the acquisitions that we made to analyse picture sharpness in function of Z values, we created several 
datasets (available at link https://app.supervisely.com/projects/265377/datasets) :
- ds0: grid course in slide.
- ds1: diagonal course stopping at 10 positions equally spaced. This diagonal represents the full diagonal on one strain 
in a slide. 
- ds2: another grid course in a slide.

Each of these datasets contains pictures taken at different XY positions and separated by a step of 5µm. Each XY 
position contains 495 pictures spaced by 5µm.
These datasets were tagged thanks to our Sober algorithm (sharpness detections) which enables to know the distance of 
the autofocus using the Z stack of pictures.












------------------

### Bibliography

[1] Deep learning-based single-shot autofocus method for digital microscopy: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8803042/

[2] Fastai — Image Regression — Age Prediction based on Image: https://medium.com/analytics-vidhya/fastai-image-regression-age-prediction-based-on-image-68294d34f2ed
