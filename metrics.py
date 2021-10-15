import numpy as np 
import matplotlib.pyplot as plt
import os 
from copy import deepcopy
import cv2

# This file will contain some functions to perform metrics on a dataset
# It will work on sard and visdrone datasets to be more precise.

# The idea is that this to be a class, and each object will be created passing a dataset, so that the object will perform metrics on a specific dataset
# Every metric object is bound to a ground truth
class Metric():
    def __init__(self,ground_truth_path,dataset_directory):
        self.dataset_directory = dataset_directory
        self.ground_truth_path = ground_truth_path
        files_in_dataset_folder = [file2 for file2 in os.listdir(self.dataset_directory)]
        # Parsing ground truth 
        ground_truth_dict = {}
        file = open(self.ground_truth_path,'r')
        for row in file.readlines():
            pieces = row.split(' ')
            if('\n' in pieces[0]):
                pieces[0] = pieces[0].replace('\n','')
            # Check if there is the corresponding image
            if(pieces[0] in files_in_dataset_folder):
                ground_truth_dict[pieces[0]]=[]
                for bbox in pieces[1:]:
                    coords = bbox.split(',')
                    try:
                        ground_truth_dict[pieces[0]].append([int(coords[0]),int(coords[1]),int(coords[2]),int(coords[3]),int(coords[4]),1])
                    except:
                        continue
                        #print('There could be a problem in the annots')
            else:
                print(pieces[0])

        self.ground_truth = ground_truth_dict
        #print(len(self.ground_truth.keys()))
        # Loading resolution dictionary
        resolution_dict = {}
        for key in self.ground_truth.keys():
            path = os.path.join(dataset_directory,key)
            temp_img = cv2.imread(path)
            resolution_dict[key] = (temp_img.shape)
        self.resolution = resolution_dict

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

        tot_g_truth = 0
        small_g_truth = 0 
        medium_g_truth = 0 
        large_g_truth = 0 
        tot_pred = 0
        # Create a copy of the ground truth to iterate 
        copy_gt_truth = deepcopy(self.ground_truth)
        # Calculating true positive
        for key in self.ground_truth.keys():
            resolution = self.resolution[key]
            for box in self.ground_truth[key]:
                area = (box[3]-box[1])*(box[2]-box[0])
                if(area<0.001*resolution[0]*resolution[1]):
                    small_g_truth+=1
                elif(area>0.01*resolution[0]*resolution[1]):
                    large_g_truth+=1
                else:
                    medium_g_truth+=1
            tot_g_truth += len(self.ground_truth[key])
            if(key in predictions_dict.keys()):
                # Creating matrix of ious
                matrix = np.zeros((len(self.ground_truth[key]),len(predictions_dict[key])))
                if(matrix.shape[0]==0 or matrix.shape[1]==0):
                        continue
                for i in range(len(self.ground_truth[key])):
                    for j in range(len(predictions_dict[key])):
                        matrix[i,j]=self.evaluate_IoU(self.ground_truth[key][i],predictions_dict[key][j])
                # iterating on the max
                max_value = np.amax(matrix)
                while(max_value>=iou_threshold):
                    max_indices = np.where(matrix == max_value)
                    true_positive+=1
                    # Check if the truth is small, medium large 
                    box = copy_gt_truth[key][max_indices[0][0]]
                    # Remove bbox from ground truth
                    del copy_gt_truth[key][max_indices[0][0]]
                    area = (box[3]-box[1])*(box[2]-box[0])
                    if(area<0.001*resolution[0]*resolution[1]):
                        small_positive+=1
                    elif(area>0.01*resolution[0]*resolution[1]):
                        large_positive+=1
                    else:
                        medium_positive+=1
                    # Deleting row and column
                    matrix = np.delete(matrix,max_indices[0][0],0)
                    matrix = np.delete(matrix,max_indices[1][0],1)
                    if(matrix.shape[0]==0 or matrix.shape[1]==0):
                        break
                    # Updating max value
                    max_value = np.amax(matrix)
        
        # Calculate total predictions    
        for key in predictions_dict.keys():
            tot_pred += len(predictions_dict[key])

        # Total precision
        if(tot_pred!=0):
            precision = true_positive/tot_pred
        else:
            precision = 0
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
        else:
            f1 = 0 

        return [precision, recall, f1, small_recall, medium_recall, large_recall]

    def calculate_precision_recall_f1_lists(self, predictions_dict, confidence_steps, iou_threshold, plot_graph=False):
        # Function that plots precision recall values for different confidences to create a curve
        # INPUT
        # predictions_list = a list of predictions for a set of images (at low confidence, for example 0.01)
        # iou_threhsold = the IOU threhsold used to quantify true positive
        precision_list = []
        recall_list = []
        f1_list = []
        rec_list_small = []
        rec_list_medium = []
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
            
            rec_list_small.append(metr_values[3])
            
            rec_list_medium.append(metr_values[4])
            
            rec_list_large.append(metr_values[5])
        
        if(plot_graph):
            plt.figure()
            plt.plot(recall_list,precision_list)
            plt.title('Precision-Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            
            plt.show()

        return precision_list, recall_list, f1_list, rec_list_small, rec_list_medium, rec_list_large

    def calc_AP(self,precision_list, recall_list):
        # Function to calculate the average precision
        # INPUT
        # precision_list = a list of floating precision values
        # recall_list = a list of floating recall values
        # OUTPUT
        # average_precision = floating value
        # This are pessimistic interpolated AP AND AR
        average_prec = 0
        # First element 
        precision_list.append(0)
        recall_list.insert(0,0)
        recall_list.append(1)
        for index in range(len(precision_list)):
            average_prec+=(recall_list[index+1]-recall_list[index])*precision_list[index]

        return average_prec

            
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
                #print('false positive')
                #print(key)
                false_positive+=1
            elif(len(truth)!=0 and len(predictions_dict[key])==0):
                #print('false negative')
                #print(key)
                false_negative+=1
                print(key)
            else:
                matrix = np.zeros((len(self.ground_truth[key]),len(predictions_dict[key])))
                for i in range(len(self.ground_truth[key])):
                    for j in range(len(predictions_dict[key])):
                        matrix[i,j]=self.evaluate_IoU(self.ground_truth[key][i],predictions_dict[key][j])
                if(np.amax(matrix)>iou):
                    # True positive 
                    true_positive+=1
                else:
                    #print('false positive')
                    #print(key)
                    false_positive+=1
        
        tot_positives = true_positive+false_negative
        tot_negatives = true_negative+false_positive
        
        print('True positive: ',(true_positive/tot_positives)*100)
        #print('False positive: ',(false_positive/tot_negatives)*100)
        #print('True negative: ',(true_negative/tot_negatives)*100)
        print('False negative: ',(false_negative/tot_positives)*100)

        fig = plt.figure()
        ax = fig.subplots()
        values = np.zeros((2,2))
        values[0][0] = (true_positive/tot_positives)*100
        values[0][1] = (false_positive/tot_negatives)*100
        values[1][1] = (true_negative/tot_negatives)*100
        values[1][0] = (false_negative/tot_positives)*100

        ax.matshow(values,cmap='Blues')
        for (i, j), z in np.ndenumerate(values):
            ax.text(j, i, str(np.round(z,2))+'%', ha='center', va='center')
        
        ax.set_xticks([0,1])
        ax.set_yticks([0,1])
        ax.set_xticklabels(['Object present','No object'])
        ax.set_yticklabels(['Object present','No object'])
        ax.xaxis.set_label_position('top')
        plt.xlabel('Truth',fontweight='bold')
        plt.ylabel('Prediction',fontweight='bold')
        plt.show()

        return true_positive, false_positive, true_negative, false_negative