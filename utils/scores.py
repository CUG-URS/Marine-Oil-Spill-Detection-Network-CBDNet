# encoding: utf-8
from sklearn.metrics import confusion_matrix  
import numpy as np
from PIL import Image
import os
import time
 
localtime = time.asctime( time.localtime(time.time()) )
def compute_recall(gt, pred):
    #  返回所有类别的召回率recall
    matrix = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten())
    recall = np.diag(matrix) / matrix.sum(axis = 0)
    return recall
	
def compute_precision(gt, pred):
    #  返回所有类别的召回率recall
	matrix = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten())
	#precision = np.diag(matrix) / matrix.sum(axis = 0)
	precision = np.diag(matrix) / matrix.sum(axis = 1)
	return precision
	
def compute_acc(gt, pred):
    matrix = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten())
    acc = np.diag(matrix).sum() / matrix.sum()
    return acc
	
def compute_kappa(prediction, target):
    prediction.tolist(), target.tolist()
    img, target = np.array(prediction).flatten(), np.array(target).flatten()
    kappa = cohen_kappa_score(target, img)
    return  kappa
def IntersectionOverUnion(gt, pred):  
	#  返回交并比IoU
	matrix = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten())
	intersection = np.diag(matrix)  
	union = np.sum(matrix, axis = 1) + np.sum(matrix, axis = 0) - np.diag(matrix)  
	IoU = intersection / union
	return IoU	
def compute_f1(gt, pred):
    confusionMatrix = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten())
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    f1 = 2 * precision * recall / (precision + recall)
    return  f1


recall_total=[0,0]
precision_total=[0,0]
acc_total=[0,0]
f1_total=[0,0]
n=0
name_truth = '/data/nancy/oil/datasets/ours_oil_spill/test/sentinel/gt/'
name_pred = '/data/nancy/oil/submits/scse_sentinel_hyjoint_loss2/'

text='\n'+name_pred

for filename in os.listdir(name_truth):              #listdir的参数是文件夹的路径
	img_truth = name_truth+'/'+filename
	img_pred = name_pred+'/'+filename
	
	im_truth = Image.open(img_truth)
	R,G,B = im_truth.split() 
	y_true = np.asarray(B)
	
	im_pred = Image.open(img_pred)
	R,G,B = im_pred.split() 
	y_pred = np.asarray(B)	

	recall = compute_recall(y_true, y_pred)
	recall_total=recall+recall_total
	precision = compute_precision(y_true, y_pred)
	precision_total=precision+precision_total
	acc = compute_acc(y_true, y_pred)
	acc_total=acc+acc_total
	f1 = compute_f1(y_true, y_pred)
	f1_total=f1+f1_total
	print(recall)
	print(precision)
	print(f1)
	print('-----------')
	n=n+1
	#print(recall)
text=text+'\nrecall_total'+str(recall_total/n)
text=text+'\nprecision_total'+str(precision_total/n)
text=text+'\nacc_total'+str(acc_total/n)
text=text+'\nf1_total'+str(f1_total/n)
text=text+'\n'+str(localtime)
with open('test.txt','a',encoding='utf-8') as f:
    f.write(text)
