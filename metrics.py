import numpy as np 
import matplotlib.pyplot as plt
import os 
from copy import deepcopy
# This file will contain some functions to perform metrics on a dataset
# It will work on sard and visdrone datasets to be more precise.

# The idea is that this to be a class, and each object will be created passing a dataset, so that the object will perform metrics on a specific dataset
# Every metric object is bound to a ground truth
class Metric():
    def __init__(self,ground_truth_path,dataset_directory):
        self.dataset_directory = dataset_directory
        self.ground_truth_path = ground_truth_path
        files_in_dataset_folder = [file for file in os.listdir(self.dataset_directory)]
        # Parsing ground truth 
        ground_truth_dict = {}
        file = open(self.ground_truth_path,'r')
        for row in file.readlines():
            pieces = row.split(' ')
            # Check if there is the corresponding image
            if(pieces[0] in files_in_dataset_folder):
                ground_truth_dict[pieces[0]]=[]
                for bbox in pieces[1:]:
                    coords = bbox.split(',')
                    ground_truth_dict[pieces[0]].append([int(coords[0]),int(coords[1]),int(coords[2]),int(coords[3]),int(coords[4]),1])
        self.ground_truth = ground_truth_dict

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
        # Create a copy of the ground truth to iterate 
        copy_gt_truth = deepcopy(self.ground_truth)
        # Calculating true positive
        for key in self.ground_truth.keys():
            if(key in predictions_dict.keys()):
                # Creating matrix of ious
                matrix = np.zeros((len(self.ground_truth[key]),len(predictions_dict[key])))
                if(matrix.shape[0]==0 or matrix.shape[1]==0):
                        continue
                for i in range(len(self.ground_truth[key])):
                    for j in range(len(predictions_dict[key])):
                        matrix[i,j]=self.evaluate_IoU(self.ground_truth[key][i],predictions_dict[key][j])
                # iterating on the max
                while(np.amax(matrix)>iou_threshold):
                    max_value = np.amax(matrix)
                    max_indices = np.where(matrix == max_value)
                    true_positive+=1
                    # Check if the truth is small, medium large 
                    box = copy_gt_truth[key][max_indices[0][0]]
                    # Remove bbox from ground truth
                    del copy_gt_truth[key][max_indices[0][0]]
                    area = (box[3]-box[1])*(box[2]-box[0])
                    if(area<256):
                        small_positive+=1
                    elif(area>1200):
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
            for box in self.ground_truth[key]:
                area = (box[3]-box[1])*(box[2]-box[0])
                if(area<256):
                    small_g_truth+=1
                elif(area>1200):
                    large_g_truth+=1
                else:
                    medium_g_truth+=1
            tot_g_truth += len(self.ground_truth[key])

        for key in predictions_dict.keys():
            for box in predictions_dict[key]:
                area = (box[3]-box[1])*(box[2]-box[0])
                if(area<256):
                    small_pred+=1
                elif(area>1200):
                    large_pred+=1
                else:
                    medium_pred+=1
            tot_g_truth += len(self.ground_truth[key])
            tot_pred += len(predictions_dict[key])


        # Total precision
        if(tot_pred!=0):
            precision = true_positive/tot_pred
        else:
            precision = 0
        # Small precision
        if(small_pred):
            small_precision = small_positive/small_pred
        else:
            small_precision = 0 
        # Medium precision
        if(medium_pred):
            medium_precision = medium_positive/medium_pred
        else:
            medium_precision = 0
        # Large precision
        if(large_pred):
            large_precision = large_positive/large_pred
        else:
            large_precision = 0 
        # Total recall
        if(tot_g_truth!=0):
            recall = true_positive/tot_g_truth
        else:
            recall = 0
        # Small recall
        if(small_g_truth!=0):
            small_recall = small_positive/small_g_truth
        else:
            small_recall = 0
        # Medium recall
        if(medium_g_truth!=0):
            medium_recall = medium_positive/medium_g_truth
        else:
            medium_recall = 0
        # Large recall
        if(large_g_truth!=0):
            large_recall = large_positive/large_g_truth
        else:
            large_recall = 0
        # Total f1
        if(precision+recall!=0):
            f1 = (2*precision*recall)/(precision+recall)
        # Small f1
        if(small_precision+small_recall!=0):
            f1_small = (2*small_precision*small_recall)/(small_precision+small_recall)
        else:
            f1_small = 0 
        # Medium f1
        if(medium_precision+medium_recall!=0):
            f1_medium = (2*medium_precision*medium_recall)/(medium_precision+medium_recall)
        else:
            f1_medium = 0 
        # Large f1
        if(large_precision+large_recall!=0):
            f1_large = (2*large_precision*large_recall)/(large_precision+large_recall)
        else:
            f1_large = 0

        return [precision, recall, f1, small_precision, small_recall, f1_small, medium_precision, medium_recall, f1_medium, large_precision, large_recall, f1_large]

    def calculate_precision_recall_f1_curve(self, predictions_dict, confidence_steps, iou_threshold, plot_graph=True):
        # Function that plots precision recall values for different confidences to create a curve
        # INPUT
        # predictions_list = a list of predictions for a set of images (at low confidence, for example 0.01)
        # iou_threhsold = the IOU threhsold used to quantify true positive
        precision_list = []
        recall_list = []
        f1_list = []
        prec_list_small = []
        rec_list_small = []
        prec_list_medium = []
        rec_list_medium = []
        prec_list_large = []
        rec_list_large = []
        list_of_predictions = []
        # Ordering the predictions by confidence in a list
        for key, value in predictions_dict.items():
            for index in range(len(value)):
                list_of_predictions.append((key,index,value[index][5])) 
        ordered_list_prediction = sorted(list_of_predictions, key=lambda x: x[2],reverse=True)
        # Creating new prediction dict, inserting one bbox at a time
        for confidence_step in confidence_steps:
            new_pred_dict = {}
            for bbox in ordered_list_prediction:
                if(bbox[0] in new_pred_dict and bbox[2]>confidence_step):
                    new_pred_dict[bbox[0]].append(predictions_dict[bbox[0]][bbox[1]])
                elif(bbox[2]>confidence_step):
                    new_pred_dict[bbox[0]] = [predictions_dict[bbox[0]][bbox[1]]]
                # Every time calculate precision recall etc
            metr_values = self.precision_recall(new_pred_dict,iou_threshold)
            precision_list.append(metr_values[0])
            recall_list.append(metr_values[1])
            f1_list.append(metr_values[2])
            
            prec_list_small.append(metr_values[3])
            rec_list_small.append(metr_values[4])
            
            prec_list_medium.append(metr_values[6])
            rec_list_medium.append(metr_values[7])
            
            prec_list_large.append(metr_values[9])
            rec_list_large.append(metr_values[10])
        
        if(plot_graph):
            plt.figure()
            plt.plot(recall_list,precision_list)
            plt.title('Precision-Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # Small objects 
            plt.figure()
            plt.plot(rec_list_small,prec_list_small)
            plt.title('Precision-Recall Curve --- Small objects')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # Medium objects 
            plt.figure()
            plt.plot(rec_list_medium,prec_list_medium)
            plt.title('Precision-Recall Curve --- Medium objects')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # Large objects 
            plt.figure()
            plt.plot(rec_list_large,prec_list_large)
            plt.title('Precision-Recall Curve --- Large objects')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            
            plt.show()

        return precision_list, recall_list, f1_list, prec_list_small, rec_list_small, prec_list_medium, rec_list_medium, prec_list_large, rec_list_large

    def calc_AP_AR(self,precision_list, recall_list):
        # Function to calculate the average precision
        # INPUT
        # precision_list = a list of floating precision values
        # recall_list = a list of floating recall values
        # OUTPUT
        # average_precision = floating value
        average_prec = 0
        average_rec = 0
        recall_list.append(1)
        for index in range(len(precision_list)):
            average_prec+=(recall_list[index+1]-recall_list[index])*precision_list[index]
        recall_list.pop()
        precision_list.append(0)
        for index in range(len(recall_list)):
            average_rec+=(precision_list[index]-precision_list[index+1])*recall_list[index]
        precision_list.pop()

        return average_prec, average_rec

            
    # NEW METRIC IMPLEMENTATION 
    def frame_metric(self, predictions_dict,iou):
        # To correctly work, it has to be high in true positive and true negative (there must be almost the same amount of empty and non empty images)
        true_positive = 0 
        true_negative = 0
        false_positive = 0 
        false_negative = 0 
        for key in self.ground_truth.keys():
            truth = self.ground_truth[key]
            if(len(truth)==0 and len(predictions_dict[key])==0):
                # True negative 
                true_negative+=1
            elif(len(truth)==0 and len(predictions_dict[key])!=0):
                # False positive
                false_positive+=1
            elif(len(truth)!=0 and len(predictions_dict[key])==0):
                false_negative+=1
            else:
                matrix = np.zeros((len(self.ground_truth[key]),len(predictions_dict[key])))
                for i in range(len(self.ground_truth[key])):
                    for j in range(len(predictions_dict[key])):
                        matrix[i,j]=self.evaluate_IoU(self.ground_truth[key][i],predictions_dict[key][j])
                if(np.amax(matrix)>iou):
                    # True positive 
                    true_positive+=1
                else:
                    false_positive+=1
        
        print('True positive: ',true_positive)
        print('False positive: ',false_positive)
        print('True negative: ',true_negative)
        print('False negative: ',false_negative)
        return true_positive, false_positive, true_negative, false_negative