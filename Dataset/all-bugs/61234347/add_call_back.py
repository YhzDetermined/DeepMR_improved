import os

import numpy
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from feature_extraction.Config import train_model_name, X_test_name, Y_test_name, params
from feature_extraction.Util.Utils import my_model_train

samples = datasets.load_iris()
X = samples.data
y = samples.target
df = pd.DataFrame(data=X)
df.columns = samples.feature_names
df['Target'] = y

# prepare data
X = df[df.columns[:-1]]
y = df[df.columns[-1]]

# hot encoding
encoder = LabelEncoder()
y1 = encoder.fit_transform(y)
y = pd.get_dummies(y1).values

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# build model
model = Sequential()
model.add(Dense(1000, activation='tanh', input_shape=((df.shape[1] - 1),)))
model.add(Dense(500, activation='tanh'))
model.add(Dense(250, activation='tanh'))
model.add(Dense(125, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(9, activation='tanh'))
model.add(Dense(y.shape[1], activation='softmax'))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_test,
    'y_val': y_test
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='categorical_crossentropy',dataset=dataset,iters=1,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_test)
numpy.save(Y_test_name+'.npy', y_test)

score, acc = model.evaluate(X_test, y_test, verbose=0)
print(score)
print(acc)
