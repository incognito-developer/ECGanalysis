import numpy as np
import pandas as pd
import tensorflow as tf
import os

#deepLearning model
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import performanceEvaluation #performanceEvaluation.py
import createModel #createModel.py

#to setup environments
gpuID = 2
baseDatasetPath = "../dataset/"
basePltSavePath = "./plots/"
baseModelPath = "./models/"
modelName="ecgTrainModel.h5"
batchsize=128


#to use gpu efficient
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[gpuID], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpuID], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)
        
def main(file1,file2,debugMode=False):
    loadModel = tf.keras.models.load_model(baseModelPath+modelName)
    data, label = loadData(file1,file2, debugMode)

    from tensorflow.keras.utils import to_categorical
    data = np.expand_dims(data,-1)

    if debugMode:
        label = to_categorical(label)
        #test_loss,test_acc = loadModel.evaluate(data,label,batch_size=batchsize,verbose=2)

        #to calculate performance Evaluation
        performanceEvaluation.performanceEvaluation(data,label,loadModel)
    else:
        print(loadModel.predict_classes(data))
        for i in range (data.shape[0]):
            #print(data[0])
            print("data ", i,"predict result: ", predictResult(loadModel.predict_classes(np.expand_dims(data[i],axis=0))), "predict: ", np.max(loadModel.predict(np.expand_dims(data[i],axis=0)))*100,"%")
        
def predictResult(predictClass):
    if predictClass == 0:
        return "normal"
    else:
        return" abnormal"
    
def loadData(file1, file2="",debugMode=False):
    data1 = ""
    if file2 != "" and debugMode:
        data1 = pd.read_csv(file1,header=None)
        data1 = np.array(data1)
        #print(data1.shape[1])
        label1 = np.split(data1,data1.shape[1],axis=1)[data1.shape[1]-1]
        data1 = np.delete(data1,data1.shape[1]-1,1)
        
        data2 = pd.read_csv(file2,header=None)
        data2 = np.array(data2)
        label2 = np.split(data2,data2.shape[1],axis=1)[data2.shape[1]-1]
        data2 = np.delete(data2,data2.shape[1]-1,1)

        data = np.concatenate((data1,data2),0)
        label = np.concatenate((label1,label2),0)

        #print("data: ",data,"label: ", label)

        return data, label

    elif file2 != "":
        data1 = pd.read_csv(file1,header=None)
        data1 = np.array(data1)
        #print(data1.shape[1])
        #label1 = np.split(data1,data1.shape[1],axis=1)[data1.shape[1]-1]
        #data1 = np.delete(data1,data1.shape[1]-1,1)
        
        data2 = pd.read_csv(file2,header=None)
        data2 = np.array(data2)
        #label2 = np.split(data2,data2.shape[1],axis=1)[data2.shape[1]-1]
        #data2 = np.delete(data2,data2.shape[1]-1,1)

        data = np.concatenate((data1,data2),0)
        #label = np.concatenate((label1,label2),0)

        #print("data: ",data,"label: ", label)

        label = ""
        return data, label

    
    elif debugMode:
        data = pd.read_csv(file1,header=None)
        data = np.array(data)
        label = np.split(data,data.shape[1],axis=1)[data.shape[1]-1]
        data = np.delete(data,data.shape[1]-1,1)

        return data,label

        
    else:
        data = pd.read_csv(file1,header=None)
        data = np.array(data)

        label = ""

        return data, label
            

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file1", dest="file1", action="store")
    parser.add_argument("--file2", dest="file2", action="store",default="")

    parser.add_argument("-d", "--debugMode", dest="debugMode", action="store", default=False, help="True: to calculate performanceEvaluation with new dataset, False: only predict about new data" )
    
    
    args = parser.parse_args()
        
    main(args.file1, args.file2, args.debugMode)
    #loadData("../dataset/ptbdb_abnormal.csv","../dataset/ptbdb_normal.csv")
