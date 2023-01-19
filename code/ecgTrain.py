import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os

import createModel #createModel.py

#useDataset: https://www.kaggle.com/shayanfazeli/heartbeat

#to setup environments
baseDatasetPath = "../dataset/"
basePltSavePath = "./plots/"
baseModelPath = "./models/"
baseCheckPointPath = baseModelPath + "checkPoints/"
batchSize = 128

#to use gpu efficient
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[2], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)


normalData = pd.read_csv(baseDatasetPath + "ptbdb_normal.csv")
print(normalData)
abnormalData = pd.read_csv(baseDatasetPath + "ptbdb_abnormal.csv")

normalData = np.array(normalData)
abnormalData = np.array(abnormalData)

print("normalData:", normalData.shape, "\n", normalData)
print("abnormalData:",abnormalData.shape, "\n", abnormalData)



#data sampling. #6:3:1=train:val:test
#trainData = normalData(3000) + abnormalData(3000) = 6000
#valData = normalData(1000) + abnormalData(1000) = 2000
#trainNumbers = 3000
#valNumbers = 1000
trainNumbers = 2400
valNumbers = 1200
testNumbers = 400


xTrain = np.concatenate((normalData[:trainNumbers,:], abnormalData[:trainNumbers,:]),0)
yTrain = np.concatenate((np.zeros(trainNumbers,),np.ones(trainNumbers,)),0) #normal = 0, abnormal = 1
xVal = np.concatenate((normalData[trainNumbers:trainNumbers+valNumbers,:], abnormalData[trainNumbers:trainNumbers+valNumbers,:]),0)
yVal = np.concatenate((np.zeros(valNumbers,), np.ones(valNumbers,)),0)
xTest = np.concatenate((normalData[trainNumbers+valNumbers:trainNumbers+valNumbers+testNumbers,:], abnormalData[trainNumbers+valNumbers:trainNumbers+valNumbers+testNumbers,:]),0)
yTest = np.concatenate((np.zeros(testNumbers,), np.ones(testNumbers,)),0)

print("xTrain.shape: ", xTrain.shape,"yTrain.shape: ", yTrain.shape)
print("xVal.shape: ", xVal.shape,"yVal.shape: ", yVal.shape)
print("xTest.shape: ", xTest.shape,"yTest.shape: ", yTest.shape)



#Converts a class vector (integers) to binary class matrix.
#to make output numbers 2
#because to match output matrix
from tensorflow.keras.utils import to_categorical
yTest = to_categorical(yTest)
yTrain = to_categorical(yTrain)
yVal = to_categorical(yVal)
xTest = np.expand_dims(xTest, -1)
xTrain = np.expand_dims(xTrain, -1)
xVal = np.expand_dims(xVal, -1)

#print(yTest)



#deepLearning model
model = createModel.createModel(xTrain) #createModel.py

# use checkpoint to save models during train
#reference: https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko
#checkpoint_path = baseCheckPointPath + "ecgTrainCheckPoint-{epoch:04d}.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, period=5)
#model.save_weights(checkpoint_path.format(epoch=0))

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience= 10)
mc = tf.keras.callbacks.ModelCheckpoint(baseCheckPointPath+'best_model.h5', monitor = 'val_acc', mode='max', verbose=1, save_best_only=True)
history=model.fit(xTrain, yTrain, epochs = 1000, batch_size=batchSize, validation_split=0.2, callbacks=[es,mc])



#plot epoch_loss
import matplotlib.pyplot as plt
tl = history.history["loss"]
vl = history.history["val_loss"]
threshold = len(history.history["val_loss"]) #to get epoch count
#print(len(history.history),history.history)
plt.plot(range(1, len(tl) + 1), tl, 'b', label="training")
plt.plot(range(1, len(vl) + 1), vl, '--r', label="validation")
plt.xlim(1, threshold)
plt.ylim(0, 1)
plt.grid(); plt.legend()
plt.xlabel('epoch'); plt.ylabel('loss')
#plt.show()
plt.savefig(basePltSavePath+"loss_epoch_plot.png")
plt.close()

ta = history.history["acc"]
va = history.history["val_acc"]
plt.plot(range(1, len(ta) + 1), ta, 'b', label="training")
plt.plot(range(1, len(va) + 1), va, '--r', label="validation")
plt.xlim(1, threshold)
plt.ylim(0, 1)
plt.grid(); plt.legend()
plt.xlabel('epoch'); plt.ylabel('accuracy')
#plt.show() 
plt.savefig(basePltSavePath+"accuracy_epoch_plot.png")
plt.close()


loss, acc = model.evaluate(xTest, yTest, batch_size=batchSize, verbose=2)
#print("loss, acc = " , loss*100, acc*100)

o = model.predict(xTest)
#print("model.predict(xTest): \n", o)

o = np.argmax(o,1) #output: 0 or 1
#print("o: ", o)

yTest = np.argmax(yTest,1)
print("yTest accuracy : ", sum(np.equal(yTest,o))/len(yTest))


model.save(baseModelPath+'ecgTrainModel.h5')
