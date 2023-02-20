#deepLearning model
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers

def createModel(xTrain):
  model = Sequential()
  dropoutRate = 0.3
  model.add(layers.Conv1D(filters=32, kernel_size=3, input_shape=(xTrain.shape[1], 1), activation='relu'))
  model.add(layers.Dropout(dropoutRate))
  model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
  model.add(layers.MaxPooling1D(pool_size = 3, strides=2))
  model.add(layers.Conv1D(filters=64, kernel_size=3, input_shape=(xTrain.shape[1],1), activation='relu'))
  model.add(layers.Dropout(dropoutRate))
  model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
  model.add(layers.MaxPooling1D(pool_size=3, strides=2))
  model.add(layers.LSTM(16))
  model.add(layers.Dense(units=2, activation="softmax"))
  #model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=0.005), metrics=['acc']) #learning_rate 0.01 -> 0.005 -> 0.001 -> 0.0005 => accuracy 0.8935 -> 0.9275 -> 0.94 -> 0.84375
  model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=0.001), metrics=['acc']) #to change dynamic learning_rate
  


  print(model.summary())
  
  return model


if __name__=="__main__":
  print("this is Module")
  createModel()
