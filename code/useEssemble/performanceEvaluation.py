
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
import numpy as np

def performanceEvaluationForEnsemble(xTest,yTest,model):
      
    yhat_probs=makeYHat(xTest,model)
    # predict crisp classes for test set
    #yhat_classes = np.argmax(model.predict(xTest), axis=-1) #multilabel
    yhat_classes = makeYHat(xTest,model)
    #yhat_classes = np.argmax(makeYHat(xTest,model), axis=0) #binary label
    #yhat_classes = (model.predict(xTest) > 0.5).astype("int32")
    #yhat_classes = model.predict_classes(xTest, verbose=0)
    #print("yhat probs : ", yhat_probs)
    #yhat_probs = np.argmax(yhat_probs,1)
    
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
    #print(yTest, yhat_classes)
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
    try:
        auc = roc_auc_score(yTest, yhat_probs)
        print('\tROC AUC: %f' % auc)
    except Exception as e:
        print('\tROC AUC: ' ,e)
    # confusion matrix
    matrix = confusion_matrix(yTest, yhat_classes)
    print("\tmatrix:\n\tPredictive Values(positive, negative)\n\tActual Values(positive, \n\t\t\tnegative)"),
    print("\t\t\t\t",matrix[0],"\n\t\t\t\t",matrix[1])

    print()

def makeYHat(xTest,models):
    yhats = [model.predict(xTest,verbose=0) for model in models]
    yhats = np.array(yhats)
    #sum across ensembles
    summed = np.sum(yhats,axis=0)
    #argmax across classes
    outcomes = np.argmax(summed,axis=1)
    #print("outputs:",outcomes.shape)
    return outcomes

def performanceEvaluation(xTest,yTest,model):
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
    try:
        auc = roc_auc_score(yTest, yhat_probs)
        print('\tROC AUC: %f' % auc)
    except Exception as e:
        print('\tROC AUC: ' ,e)
    # confusion matrix
    matrix = confusion_matrix(yTest, yhat_classes)
    print("\tmatrix:\n\tPredictive Values(positive, negative)\n\tActual Values(positive, \n\t\t\tnegative)"),
    print("\t\t\t\t",matrix[0],"\n\t\t\t\t",matrix[1])

    print()

if __name__ =="__main__":
    print("this is Module")
    performanceEvaluation()
