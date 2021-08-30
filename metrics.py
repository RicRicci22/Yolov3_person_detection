import os
from tool import utils
import numpy as np 
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

    def precision_recall(self, predictions_dict,iou_threshold):
        # iou_threshold = the threshold to define a true positive
        # predicted_annotations = dictionary with key = image name, value = list of list [[bbox1][bbox2][bbox3]] of predicted boxes
        true_positive = 0
        small_positive = 0
        medium_positive = 0 
        large_positive = 0
        # Calculating true positive
        for key in self.ground_truth.keys():
            if(key in predictions_dict.keys()):
                # Creating matrix of ious
                matrix = np.zeros((len(self.ground_truth[key]),len(predictions_dict)))
                for i in  range(len(self.ground_truth[key])):
                    for j in range(len(predictions_dict[key])):
                        matrix[i,j]=self.evaluate_IoU(self.ground_truth[key][i],predictions_dict[key][j])
                # iterating on the max
                while(np.amax(matrix)>iou_threshold):
                    max_value = np.amax(matrix)
                    max_indices = np.where(matrix == max_value)
                    true_positive+=1
                    # Check if the truth is small, medium large 
                    box = self.ground_truth[key][max_indices[0][0]]
                    #print(bbox_truth)
                    area = (box[3]-box[1])*(box[2]-box[0])
                    if(area<150):
                        small_positive+=1
                    elif(area>500):
                        large_positive+=1
                    else:
                        medium_positive+=1
                    # Deleting row and column
                    matrix = np.delete(matrix,max_indices[0][0],0)
                    matrix = np.delete(matrix,max_indices[1][0],1)
                    if(matrix.shape[0]==0 or matrix.shape[1]==0):
                        break
    
        # Calculate total number of ground truth boxes and total number of predictions
        tot_g_truth = 0
        small_g_truth = 0 
        medium_g_truth = 0 
        large_g_truth = 0 
        tot_pred = 0
        small_pred = 0 
        medium_pred = 0 
        large_pred = 0 
        f1 = 0 
        f1_small = 0
        f1_medium = 0 
        f1_large = 0 

        for key in self.ground_truth.keys():
            # So I can take also a subset of a dataset without modifying annotations
            if(key in predictions_dict.keys()):
                for box in self.ground_truth[key]:
                    area = (box[3]-box[1])*(box[2]-box[0])
                    if(area<150):
                        small_g_truth+=1
                    elif(area>500):
                        large_g_truth+=1
                    else:
                        medium_g_truth+=1
                tot_g_truth += len(self.ground_truth[key])

        for key in predictions_dict.keys():
            for box in predictions_dict[key]:
                    area = (box[3]-box[1])*(box[2]-box[0])
                    if(area<150):
                        small_pred+=1
                    elif(area>500):
                        large_pred+=1
                    else:
                        medium_pred+=1
            tot_pred += len(predictions_dict[key])


        # Calculate total precision and recall
        if(tot_pred!=0):
            precision = true_positive/tot_pred
            # DOES NOT HAVE MUCH SENSE
            # small_precision = small_positive/tot_pred
            # medium_precision = medium_positive/tot_pred
            # large_precision = large_positive/tot_pred
        else:
            precision = 0
            # small_precision = 0
            # medium_precision = 0 
            # large_precision = 0
        
        if(tot_g_truth!=0):
            recall = true_positive/tot_g_truth
            small_recall = small_positive/small_g_truth
            medium_recall = medium_positive/medium_g_truth
            large_recall = large_positive/large_g_truth
        else:
            recall = 0
            small_recall = 0
            medium_recall = 0 
            large_recall = 0 

        if(precision+recall!=0):
            f1 = (2*precision*recall)/(precision+recall)
        
        # if(small_precision+small_recall!=0):
        #     f1_small = (2*small_precision*small_recall)/(small_precision+small_recall)
        # if(medium_precision+medium_recall!=0):
        #     f1_medium = (2*medium_precision*medium_recall)/(medium_precision+medium_recall)
        # if(large_precision+large_recall!=0):
        #     f1_large = (2*large_precision*large_recall)/(large_precision+large_recall)


        return [precision, recall, f1, small_pred, small_recall, medium_pred, medium_recall, large_pred, large_recall]

    def precision_recall_scales(self,predictions_annotations,iou_threhsold, small_threshold, large_threshold):
        # Function that separates the precision and recall between small, medium and large objects
        # Small : area < 150 pixels
        # Medium : 150 < area < 500 pixels
        # Large: area > 500 pixels
        small_dictionary = {}
        medium_dictionary = {}
        large_dictionary = {}
        small_objs = 0
        medium_objs = 0
        large_objs = 0 
        for key in self.ground_truth.keys():
            small_dictionary[key] = []
            medium_dictionary[key] = []
            large_dictionary[key] = []
            boxes = predictions_annotations[key]
            for box in boxes:
                width = box[3]-box[1]
                height = box[2]-box[0]
                if(width*height<=small_threshold):
                    small_dictionary[key].append(box)
                    small_objs+=1
                elif(width*height>=large_threshold):
                    large_dictionary[key].append(box)
                    large_objs+=1
                else:
                    medium_dictionary[key].append(box)
                    medium_objs+=1
        print('# Small objects: '+str(small_objs))
        print('# Medium objects: '+str(medium_objs))
        print('# Large objects: '+str(large_objs))

        small_values = self.precision_recall(small_dictionary,iou_threhsold)
        medium_values = self.precision_recall(medium_dictionary,iou_threhsold)
        large_values = self.precision_recall(large_dictionary,iou_threhsold)

        return small_values, medium_values, large_values


    def calculate_precision_recall_lists(self, predictions_list, iou_threshold):
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


