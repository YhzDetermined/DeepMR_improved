import os

from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.datasets import make_classification

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train
import numpy
X, y = make_classification(1000, 283, n_classes=4, n_informative=4)
y = to_categorical(y)

model = Sequential()
model.add(Dense(100, input_dim=283, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(150, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, kernel_initializer='normal', activation='sigmoid'))
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# model.fit(X, y, epochs=5)
dataset = {
    'x': X,
    'y': y,
    'x_val': X,
    'y_val': y
}
check_interval = 'epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model, optimizer=sgd, loss='categorical_crossentropy', dataset=dataset, iters=5, batch_size=32,
               callbacks=[], verb=1, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', y)