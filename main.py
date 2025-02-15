import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


dataset = pd.read_csv('cancer.csv')

#The drop() method removes columns that you don't need. In this case, the drop() is removing the column that contains the target variable ('diagnosis(1=m, 0=b)'), which is the column we are trying to predict.
x = dataset.drop([col.strip() for col in dataset.columns if 'diagnosis' in col], axis=1)
y = dataset['diagnosis(1=m, 0=b)']  # Assuming this is the target column

#Splits the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Defines model
model = tf.keras.models.Sequential()

#Dense is a fully connected neural network layer, 3 layers, sigmoid for binary classification(probability)
model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification

#Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Train the model 1000 times
model.fit(x_train, y_train, epochs=1000)

#returns the loss and accurary of the model, from tests the accuracy will be around 97%
model.evaluate(x_test, y_test)