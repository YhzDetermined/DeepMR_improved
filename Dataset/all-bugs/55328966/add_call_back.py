import os

import numpy
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.regularizers import l2

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

batch_size = 32
epochs = 10
alpha = 0.0001
lambda_ = 0
h1 = 50

train = pd.read_csv('mnist_train.csv.zip')
test = pd.read_csv('mnist_test.csv.zip')

train = train.loc['1':'5000', :]
test = test.loc['1':'2000', :]

train = train.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)

x_train = train.loc[:, '1x1':'28x28']
y_train = train.loc[:, 'label']

x_test = test.loc[:, '1x1':'28x28']
y_test = test.loc[:, 'label']

x_train = x_train.values
y_train = y_train.values

x_test = x_test.values
y_test = y_test.values

nb_classes = 10
targets = y_train.reshape(-1)
y_train_onehot = np.eye(nb_classes)[targets]

nb_classes = 10
targets = y_test.reshape(-1)
y_test_onehot = np.eye(nb_classes)[targets]

model = Sequential()
model.add(Dense(784, input_shape=(784,)))
model.add(Dense(h1, activation='relu', kernel_regularizer=l2(lambda_)))
model.add(Dense(10, activation='sigmoid', kernel_regularizer=l2(lambda_)))
optimizer = SGD(learning_rate=alpha)
# model.compile(optimizer=GradientDescentOptimizer(alpha),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train_onehot, epochs=epochs, batch_size=batch_size)
dataset = {
    'x': x_train,
    'y': y_train_onehot,
    'x_val': x_test,
    'y_val': y_test_onehot
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=optimizer,loss='categorical_crossentropy',dataset=dataset,iters=epochs,batch_size=batch_size,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', x_test)
numpy.save(Y_test_name+'.npy', y_test_onehot)