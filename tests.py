from detection import Detector
from metrics import Metric
from models import Yolov4
from tool.utils import *
import matplotlib.pyplot as plt


# This module is meant to systematically process tests on dataset
# RESOLUTION TEST
# This test is meant to inspect how the input resolution influeces the detection performance
# TEST 1
# Parameters:
# Pretrained_weights
resolutions = [416,512,608,704,800,896,992,1088]
anno_path = r'datasets\sard\test\_annotations.txt'
dataset_path = r'datasets\sard\test'
# Creating the model
model = Yolov4(yolov4conv137weight=None,n_classes=80,inference=True)
model.load_weights(r'C:\Users\Melgani\Desktop\master_degree\weight\yolov4.pth')
model.activate_gpu()
# Creating element to measure metrics
metrica = Metric(anno_path,dataset_path)
# Creating file to store results
file = open(r'tests\input_resolution\sard\pretrained\no_keep_aspect_ratio\test.txt','w')
iou_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
confidence_steps = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.20,0.15,0.10,0.05]
for resolution in resolutions:
    average_prec_list = []
    average_rec_list = []
    print('Calculating for resolution '+str(resolution)+'x'+str(resolution))
    # Creating the detector
    yolov4_detector = Detector(model,True,resolution,resolution,dataset_path,keep_aspect_ratio=False)
    # Get the predictions with a low confidence 
    predictions_dict = yolov4_detector.detect_in_images(0.01,False,False)
    precision_matrix = np.zeros((len(iou_list),len(confidence_steps)))
    recall_matrix = np.zeros((len(iou_list),len(confidence_steps)))
    for index_iou in range(len(iou_list)):
        print('Calculating for iou threshold '+str(iou_list[index_iou]))
        values = metrica.calculate_precision_recall_f1_curve(predictions_dict,confidence_steps,iou_list[index_iou],plot_graph=False)
        # Calculating average precision and recall
        file.write(str(resolution)+'x'+str(resolution)+'\n')
        file.write('Iou threshold for true positive: '+str(iou_list[index_iou])+'\n')
        average_prec, average_rec = metrica.calc_AP_AR(values[0],values[1])
        average_prec_list.append(average_prec)
        average_rec_list.append(average_rec)
        # Insert in precision and recall matrices 
        precision_matrix[index_iou,:] = values[0]
        recall_matrix[index_iou,:] = values[1]
        #plt.plot(values[1],values[0],label='IoU '+str(iou))
        file.write('Average precision: '+str(np.around(average_prec,2))+'\n')
        file.write('Average recall: '+str(np.around(average_rec,2))+'\n\n')
    # Plot matrices 
    fig = plt.figure(figsize=(10,5))
    ax = fig.subplots()
    ax.matshow(precision_matrix, cmap=plt.cm.Blues)
    # Displaying text 
    for i in range(precision_matrix.shape[0]):
        for j in range(precision_matrix.shape[1]):
            c = precision_matrix[i,j]
            ax.text(j, i, str(np.around(c,2)), va='center', ha='center')
    ax.set_xticks(np.arange(len(confidence_steps)))
    ax.set_yticks(np.arange(len(iou_list)))
    ax.set_xticklabels(confidence_steps)
    ax.set_yticklabels(iou_list)
    ax.set_xlabel('Confidences threshold in prediction')
    ax.set_ylabel('IoU threshold for detection')
    ax.set_title('PRECISION VALUES for different parameters settings')
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    fig.tight_layout()
    # Save figure
    plt.savefig(os.path.join(r"C:\Users\Melgani\Desktop\master_degree\tests\input_resolution\sard\pretrained\no_keep_aspect_ratio","precision_at_"+str(resolution)+"x"+str(resolution)))


    fig2 = plt.figure(figsize=(10,5))
    ax2 = fig2.subplots()
    ax2.matshow(recall_matrix, cmap=plt.cm.Blues)
    # Displaying text
    for i in range(recall_matrix.shape[0]):
        for j in range(recall_matrix.shape[1]):
            c = recall_matrix[i,j]
            ax2.text(j, i, str(np.around(c,2)), va='center', ha='center')
    ax2.set_xticks(np.arange(len(confidence_steps)))
    ax2.set_yticks(np.arange(len(iou_list)))
    ax2.set_xticklabels(confidence_steps)
    ax2.set_yticklabels(iou_list)
    ax2.set_xlabel('Confidences threshold in prediction')
    ax2.set_ylabel('IoU threshold for detection')
    ax2.set_title('RECALL VALUES for different parameters settings')
    ax2.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    fig2.tight_layout()
    # Save figure
    plt.savefig(os.path.join(r"C:\Users\Melgani\Desktop\master_degree\tests\input_resolution\sard\pretrained\no_keep_aspect_ratio","recall_at_"+str(resolution)+"x"+str(resolution)))

    fig3 = plt.figure(figsize=(5,5))
    ax3 = fig3.subplots()
    ax3.plot(np.arange(len(iou_list)),average_prec_list)
    ax3.set_xticks(np.arange(len(iou_list)))
    ax3.set_xticklabels(iou_list)
    ax3.set_xlabel('IoU threshold for detection')
    ax3.set_ylabel('Average precision')
    ax3.set_title('AVERAGE PRECISION VS IOU')
    fig3.tight_layout()
    # Save figure
    plt.savefig(os.path.join(r"C:\Users\Melgani\Desktop\master_degree\tests\input_resolution\sard\pretrained\no_keep_aspect_ratio","average_precision_at_"+str(resolution)+"x"+str(resolution)))

    fig4= plt.figure(figsize=(5,5))
    ax4 = fig4.subplots()
    ax4.plot(np.arange(len(iou_list)),average_rec_list)
    ax4.set_xticks(np.arange(len(iou_list)))
    ax4.set_xticklabels(iou_list)
    ax4.set_xlabel('IoU threshold for detection')
    ax4.set_ylabel('Average recall')
    ax4.set_title('AVERAGE RECALL VS IOU')
    fig4.tight_layout()
    # Save figure
    plt.savefig(os.path.join(r"C:\Users\Melgani\Desktop\master_degree\tests\input_resolution\sard\pretrained\no_keep_aspect_ratio","average_recall_at_"+str(resolution)+"x"+str(resolution)))

    plt.close('all')

file.close()