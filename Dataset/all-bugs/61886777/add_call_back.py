import os

import numpy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.datasets import make_classification

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X_train, y_train = make_classification(1000, 28 * 28, n_classes=3, n_informative=5)
X_train = X_train.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train)
input_shape = (28, 28, 1)

model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# more conv layers

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

# model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#
# model.fit(X_train, y_train, epochs=5)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_train,
    'y_val': y_train
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=Adam(),loss='categorical_crossentropy',dataset=dataset,iters=5,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_train)
numpy.save(Y_test_name+'.npy', y_train)
