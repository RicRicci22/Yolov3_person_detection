import os

# This file will contain some functions to perform metrics on a dataset 
# It will work on sard and visdrone datasets to be more precise. 

def IoU_dataset(ground_truth_annotations, predictions_annotations):
    # prediction must be the same format as ground_truth annotations. 
    # Parsing ground truth:
    g_truth = open(ground_truth_annotations,'r')
    map_g_truth = {}
    for line in g_truth.readlines():
        pieces = line.split(' ')
        map_g_truth[pieces[0]] = pieces[1:]
    predictions = open(predictions_annotations,'r')
    map_predictions = {}
    for line in predictions.readlines():
        pieces = line.split(' ')
        map_predictions[pieces[0]] = pieces[1:]
    # Calculating IoU over bounding boxes and estimating the metrics
    