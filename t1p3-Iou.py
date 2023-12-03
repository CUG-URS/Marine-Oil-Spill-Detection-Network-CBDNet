# encoding: utf-8

from sklearn.metrics import confusion_matrix  
import numpy as np
from PIL import Image
import os
olderr = np.seterr(all='ignore')
name_truth = '.datasets/test/palsar/gt/'
name_pred = '.submits/palsar_CBDNet/'
mylog = open('.datasets/test/palsar'+'.log','w')

def compute_iou1(img_truth,img_pred):
    # ytrue, ypred is a flatten vector
    im_truth = Image.open(img_truth)
    R,G,B = im_truth.split() 
    y_true = np.asarray(R)
	
    im_pred = Image.open(img_pred)
    R,G,B = im_pred.split() 
    y_pred = np.asarray(R)	
	
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=[255,0])
    #print >> mylog, current
    TP = current[1][1]*1.0
    FN = current[1][0]*1.0
    TN = current[0][0]*1.0
    FP = current[0][1]*1.0
    #Accuracy = (TP + TN)/512/512
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    accuracy = (TP+TN)/ (TP+FP+TN+FN)
    fscore = 2 * (recall * precision)/ (recall + precision)
    # compute mean iou
    
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = (ground_truth_set + predicted_set - intersection).astype(np.float32)
    Iou = intersection/union
    mIoU = np.nanmean(Iou) # 求各类别IoU的平均


    return (TP,FN,TN,FP,intersection,union.astype(np.float32))



    
TP_total = 0
FN_total = 0
TN_total = 0
FP_total = 0
intersection_total = 0
union_total = 0

print(mylog, '文件名，able，Iou_1，Iou_2，mIoU，fscore，recall，precision, accuracy')
for filename in os.listdir(name_truth):              #listdir的参数是文件夹的路径
    img_truth = name_truth+filename
    img_pred = name_pred+filename

    TP,FN,TN,FP,intersection,union = compute_iou1(img_truth,img_pred)
    
    precision1 = TP / (TP+FP)
    recall1 = TP / (TP+FN)
    accuracy1 = (TP+TN)/ (TP+FP+TN+FN)
    fscore1 = 2 * (recall1 * precision1)/ (recall1 + precision1)
    Iou1 = intersection/(union)
    mIoU1 = np.nanmean(Iou1)
    
    TP_total = TP_total+TP
    FN_total = FN_total+FN
    TN_total = TN_total+TN
    FP_total = FP_total+FP

    intersection_total = intersection_total+intersection
    union_total = union_total+union

    
    precision = TP_total / (TP_total+FP_total)
    recall = TP_total / (TP_total+FN_total)
    accuracy = (TP_total+TN_total)/ (TP_total+FP_total+TN_total+FN_total)
    fscore = 2 * (recall * precision)/ (recall + precision)
    Iou = intersection_total/(union_total)
    mIoU = np.nanmean(Iou)

    
    #print (str(filename)+' '+str(round(mIoU,6))+' '+str(round(fscore,6))+' '+str(round(recall,6))+' '+str(round(precision,6))+' '+str(round(accuracy,6)))

    print(mylog,str(filename)+' '+str(round(Iou1[0],6))+' '+str(round(Iou1[1],6))+' '+str(round(mIoU1,6))+' '+str(round(fscore1,6))+' '+str(round(recall1,6))+' '+str(round(precision1,6))+' '+str(round(accuracy1,6)))

print( mylog, '--------------------')	
print(mylog,'总mIou：'+str(Iou[0])) 
print(mylog,'总mIou：'+str(Iou[1]))
print(mylog,'总mIou：'+str(mIoU))
print(mylog,'总fscore：'+str(fscore))
print(mylog,'总recall：'+str(recall)) 
print(mylog,'总precision'+str(precision))
print(mylog,'总accuracy'+str(accuracy))
print(mylog, 'Finish!')
print('Finish!') 
mylog.close()