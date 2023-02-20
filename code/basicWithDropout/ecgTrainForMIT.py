import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os

import createModel #createModel.py
import performanceEvaluation #performanceEvaluation.py

#useDataset: https://www.kaggle.com/shayanfazeli/heartbeat

#to setup environments
gpuID = 2
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
    tf.config.experimental.set_visible_devices(gpus[gpuID], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpuID], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)


#normalData = pd.read_csv(baseDatasetPath + "ptbdb_normal.csv",header=None)
#print(normalData)
#abnormalData = pd.read_csv(baseDatasetPath + "ptbdb_abnormal.csv",header=None)
data = pd.read_csv(baseDatasetPath+"mitbih_train.csv",header=None)
for i in range (data.shape[0]):
  if data.loc[i,data.columns[data.shape[1]-1]] != 0:
    data.loc[i,data.columns[data.shape[1]-1]]= 1

#to make data balance
data = data.groupby(data.columns[data.shape[1]-1]).head(min(data[data.columns[data.shape[1]-1]].value_counts()))
print("Data shape: ", data.shape)

data = np.array(data)

label = np.split(data,data.shape[1],axis=1)[data.shape[1]-1]
#label[(label > 0)]=1
#print(label)
data = np.delete(data,data.shape[1]-1,1)

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(data, label, test_size=0.10)



print("data.shape: ", data.shape,"label.shape: ", label.shape)


#Converts a class vector (integers) to binary class matrix.
#to make output numbers 2
#because to match output matrix
from tensorflow.keras.utils import to_categorical
yTest = to_categorical(yTest)
yTrain = to_categorical(yTrain)
#yVal = to_categorical(yVal)
#print(xTest)
xTest = np.expand_dims(xTest, -1)
xTrain = np.expand_dims(xTrain, -1)
#xVal = np.expand_dims(xVal, -1)

print(xTrain.shape, yTrain.shape)



#print("xTest: ", xTest,"yTest: ",yTest)

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
history=model.fit(xTrain, yTrain, epochs = 10000, batch_size=batchSize, validation_split=0.3, callbacks=[es,mc])



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
plt.grid()
plt.legend()
plt.xlabel('epoch'); plt.ylabel('accuracy')
#plt.show() 
plt.savefig(basePltSavePath+"accuracy_epoch_plot.png")
plt.close()


loss, acc = model.evaluate(xTest, yTest, batch_size=batchSize, verbose=2)
print("loss, acc = " , loss*100, acc*100)

'''
o = model.predict(xTest)
#print("model.predict(xTest): \n", o)

o = np.argmax(o,1) #output: 0 or 1
#print("o: ", o)

yTest = np.argmax(yTest,1)

#print("\n!!!WARNING: accuracy could change with same data because of dropout layer!!!") #this is wrong. please look https://stats.stackexchange.com/questions/569871/neural-network-gives-very-different-accuracies-if-repeated-on-same-data-why
print("yTest")
print("\taccuracy : ", sum(np.equal(yTest,o))/len(yTest))
'''

#to calculate performance Evaluation
performanceEvaluation.performanceEvaluation(xTest,yTest,model)

'''
#to get performance evaluation: https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/
# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
#from keras.models import Sequential
#from keras.layers import Dense

# predict probabilities for test set
yhat_probs = model.predict(xTest, verbose=0)
# predict crisp classes for test set
#yhat_classes = np.argmax(model.predict(xTest), axis=-1) #multilabel
yhat_classes = np.argmax(model.predict(xTest), axis=1) #binary label
#yhat_classes = (model.predict(xTest) > 0.5).astype("int32")
#yhat_classes = model.predict_classes(xTest, verbose=0)
#print("yhat probs : ", yhat_probs)
yhat_probs = np.argmax(yhat_probs,1)
#print("yhat_probs: ",yhat_probs)
#print("yhat_classes: ",yhat_classes)
yTest = np.argmax(yTest,1) #to make binary label
#yTest = yTest.astype(np.int64)
#print(yhat_classes)
#print("yTest: ", yTest.dtype)

print("\n!!!WARNING: accuracy could change with same data. Please look https://stats.stackexchange.com/questions/569871/neural-network-gives-very-different-accuracies-if-repeated-on-same-data-why ")
print("\n!!!WARNING: accuracy should not always match between model.evaluate VS sklearn.metrics.accuracy_score. Please look  https://datascience.stackexchange.com/questions/13920/accuracy-doesnt-match-in-keras/14500#14500")
print("yTest")
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(yTest, yhat_classes)
print('\tAccuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(yTest, yhat_classes,average='binary')
print('\tPrecision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(yTest, yhat_classes,average='binary')
print('\tRecall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(yTest, yhat_classes,average='binary')
print('\tF1 score: %f' % f1)
# kappa
kappa = cohen_kappa_score(yTest, yhat_classes)
print('\tCohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(yTest, yhat_probs)
print('\tROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(yTest, yhat_classes)
print("\tmatrix:\n\tPredictive Values(positive, negative)\n\tActual Values(positive, \n\t\t\tnegative)"),
print("\t\t\t\t",matrix[0],"\n\t\t\t\t",matrix[1])
'''
model.save(baseModelPath+'ecgTrainModel.h5')
