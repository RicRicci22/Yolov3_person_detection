import os
from tool import utils
# This file will contain some functions to perform metrics on a dataset
# It will work on sard and visdrone datasets to be more precise.

# The idea is that this to be a class, and each object will be created passing a dataset, so that the object will perform metrics on a specific dataset
# Every metric object is bound to a ground truth and a detector
class Metric():
    def __init__(self,ground_truth_path,ground_truth_dictionary,detector):
        self.ground_truth_path = ground_truth_path
        self.ground_truth = ground_truth_dictionary
        self.detector = detector

    def __str__(self):
        print('Metric object')
        print('Associated detector:\n\n',self.detector)

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

    def precision_recall(self, predictions_annotations,threshold):
        # ground_truth_annotations = dictionary with key = image name, value = list of list [[bbox1][bbox2][bbox3]] of g.t. boxes
        # predicted_annotations = dictionary with key = image name, value = list of list [[bbox1][bbox2][bbox3]] of predicted boxes
        # Calculating IoU over bounding boxes and estimating the metrics
        true_positive = 0
        # Calculating true positive
        for key in self.ground_truth.keys():
            if(key in predictions_annotations.keys()):
                for box in  self.ground_truth[key]:
                    for box2 in predictions_annotations[key]:
                        if(self.evaluate_IoU(box,box2)>threshold):
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

        return precision, recall, f1

    def calculate_ap():
        print('Under construction')


    def plot_precision_recall_curve(self, conf_step, save_plot=False, save_path=None):
        import matplotlib.pyplot as plt
        # Function that plots precision recall values for different confidences to create a curve
        # INPUT
        # ground_truth_dict
        # conf_step = step of confidence to use to calculate confidence values in this way conf_values = [0:conf_step:1]
        # save_plot = if True the function will save the precision recall curve in the specified path
        # save_path = path where to save the plot
        # OUTPUT
        # Nothing
        conf_values = [i/100 for i in range(1,100,int(conf_step*100))]
        conf_values.append(0.99)
        precision_list = []
        recall_list = []
        for conf in range(1,100,int(conf_step*100)):
            # Perform detection and get precision and recall
            predictions = self.detector.detect_in_images_pytorch(conf/100)
            precision, recall, _ = self.precision_recall(predictions,0.5)
            if(precision==0 and recall==0):
                break
            else:
                precision_list.append(precision)
                recall_list.append(recall)

        plt.plot(precision_list,recall_list)
        plt.title('Precision-Recall Curve')
        plt.show()
        return