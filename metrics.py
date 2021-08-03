import os

# This file will contain some functions to perform metrics on a dataset
# It will work on sard and visdrone datasets to be more precise.

def evaluate_IoU(coords_predicted, coords_ground_truth):
    # Return the IoU value
    # coords_predicted = [x1,y1,x2,y2] predicted
    # coords_ground_truth = [x1,y1,x2,y2] ground truth
    # GET THE INTERSECTION
    x_left = max(coords_predicted[0],coords_ground_truth[0])
    y_top = max(coords_predicted[1], coords_ground_truth[1])
    x_right = min(coords_predicted[2], coords_ground_truth[2])
    y_bottom = min(coords_predicted[3], coords_ground_truth[3])
    # Asserting if there is overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right-x_left)*(y_bottom-y_top)

    area_predicted = (coords_predicted[2]-coords_predicted[0])*(coords_predicted[3]-coords_predicted[1])
    area_gtruth = (coords_ground_truth[2]-coords_ground_truth[0])*(coords_ground_truth[3]-coords_ground_truth[1])

    return (intersection)/(area_gtruth+area_predicted-intersection)

def precision_recall(ground_truth_annotations, predictions_annotations,threshold):
    # ground_truth_annotations = dictionary with key = image name, value = list of list [[bbox1][bbox2][bbox3]] of g.t. boxes
    # predicted_annotations = dictionary with key = image name, value = list of list [[bbox1][bbox2][bbox3]] of predicted boxes
    keys_g_truth = [key for key in ground_truth_annotations.keys()]
    keys_prediction = [key for key in predictions_annotations.keys()]
    # Calculating IoU over bounding boxes and estimating the metrics
    true_positive = 0
    # Calculating true positive
    for key in keys_prediction:
        for box in predictions_annotations[key]:
            list_iou = [evaluate_IoU(box,box2) for box2 in ground_truth_annotations[key]]
            temp_list = [1 if iou>threshold else 0 for iou in list_iou]
            if(sum(temp_list)>0):
                # True positive
                true_positive += 1

    # Calculate total number of ground truth boxes and total number of predictions
    tot_g_truth = 0
    tot_pred = 0

    for key in ground_truth_annotations.keys():
        tot_g_truth += len(ground_truth_annotations[key])

    for key in predictions_annotations.keys():
        tot_pred += len(predictions_annotations[key])


    # Calculate precision and recall
    if(tot_pred!=0):
        precision = true_positive/tot_pred
    else:
        precision = 0
    if(tot_g_truth!=0):
        recall = true_positive/tot_g_truth
    else:
        recall = 0

    f1 = (2*precision*recall)/(precision+recall)

    return precision, recall, f1
