import os
from tool import utils
# This file will contain some functions to perform metrics on a dataset
# It will work on sard and visdrone datasets to be more precise.

# The idea is that this to be a class, and each object will be created passing a dataset, so that the object will perform metrics on a specific dataset
# Every metric object is bound to a ground truth
class Metric():
    def __init__(self,ground_truth_path,ground_truth_dictionary):
        self.ground_truth_path = ground_truth_path
        self.ground_truth = ground_truth_dictionary

    def __str__(self):
        print('Metric object')

        return '\nGround truth path: '+str(self.ground_truth_path)

    def evaluate_IoU(self, coords_predicted, coords_ground_truth):
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

    def precision_recall(self, predictions_annotations,iou_threshold):
        # ground_truth_annotations = dictionary with key = image name, value = list of list [[bbox1][bbox2][bbox3]] of g.t. boxes
        # iou_threshold = the threshold to define a true positive
        # predicted_annotations = dictionary with key = image name, value = list of list [[bbox1][bbox2][bbox3]] of predicted boxes
        # Calculating IoU over bounding boxes and estimating the metrics
        true_positive = 0
        # Calculating true positive
        for key in self.ground_truth.keys():
            if(key in predictions_annotations.keys()):
                for box in  self.ground_truth[key]:
                    for box2 in predictions_annotations[key]:
                        if(self.evaluate_IoU(box,box2)>iou_threshold):
                            true_positive += 1
                            break


        # Calculate total number of ground truth boxes and total number of predictions
        tot_g_truth = 0
        tot_pred = 0

        for key in self.ground_truth.keys():
            # So I can take also a subset of a dataset without modifying annotations
            if(key in predictions_annotations.keys()):
                tot_g_truth += len(self.ground_truth[key])

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

        if(precision+recall!=0):
            f1 = (2*precision*recall)/(precision+recall)
        else:
            f1 = 0

        return [precision, recall, f1]

    def calculate_precision_recall_small_medium_large(self,predictions_annotations,iou_threhsold, small_threshold, large_threshold):
        # Function that separates the precision and recall between small, medium and large objects
        # Small : area < 150 pixels
        # Medium : 150 < area < 500 pixels
        # Large: area > 500 pixels
        small_dictionary = {}
        medium_dictionary = {}
        large_dictionary = {}
        for key in predictions_annotations.keys():
            small_dictionary[key] = []
            medium_dictionary[key] = []
            large_dictionary[key] = []
            boxes = predictions_annotations[key]
            for box in boxes:
                width = box[3]-box[1]
                height = box[2]-box[0]
                if(width*height<=small_threshold):
                    small_dictionary[key].append(box)
                elif(width*height>=large_threshold):
                    large_dictionary[key].append(box)
                else:
                    medium_dictionary[key].append(box)

        small_values = self.precision_recall(small_dictionary,iou_threhsold)
        medium_values = self.precision_recall(medium_dictionary,iou_threhsold)
        large_values = self.precision_recall(large_dictionary,iou_threhsold)

        return small_values, medium_values, large_values


    def calculate_precisio_recall_lists(self, predictions_list, iou_threshold):
        # Function that plots precision recall values for different confidences to create a curve
        # INPUT
        # predictions_list = a list of predictions, one at each threhsold
        # iou_threhsold = the IOU threhsold used to quantify true positive
        precision_list = []
        recall_list = []
        for prediction in predictions_list:
            values = self.precision_recall(prediction,iou_threshold)
            if(values[0]==0 and values[1]==0):
                break
            else:
                precision_list.append(values[0])
                recall_list.append(values[1])
        return precision_list, recall_list



    def plot_precision_recall_curve(self, precision_list, recall_list, save_plot=False, save_path=None):
        import matplotlib.pyplot as plt
        # Function that plots precision recall values for different confidences to create a curve
        # INPUT
        # precision_list = list of precision values
        # recall_list = list of recall values
        # save_plot = if True the function will save the precision recall curve in the specified path
        # save_path = path where to save the plot
        plt.plot(precision_list,recall_list)
        plt.title('Precision-Recall Curve')
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.show()
        return

    def calc_average_precision(self,precision_list, recall_list):
        # Function to calculate the average precision
        # INPUT
        # precision_list = a list of floating precision values
        # recall_list = a list of floating recall values
        # OUTPUT
        # average_precision = floating value
        recall_list.append(0.0)
        precision_list.append(1.0)
        recall_list.reverse()
        precision_list.reverse()
        AP = 0
        for index in range(len(recall_list)-1):
            AP+=(recall_list[index+1]-recall_list[index])*precision_list[index]

        return AP


