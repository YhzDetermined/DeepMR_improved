import os

import numpy
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_classification(1000, 25)
X = X.reshape(-1, 5, 5, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

start_cnn = Sequential()

start_cnn.add(Conv2D(32, (3, 3), input_shape=(5, 5, 1), activation='relu', padding='same'))
start_cnn.add(Conv2D(32, (3, 3), activation='relu'))
start_cnn.add(MaxPooling2D(padding='same'))

for i in range(0, 2):
    start_cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

start_cnn.add(MaxPooling2D(padding='same'))

for i in range(0, 2):
    start_cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

start_cnn.add(MaxPooling2D(padding='same'))

# Flattening
start_cnn.add(Flatten())

# Step 4 - Full connection
start_cnn.add(Dense(activation="relu", units=128))
start_cnn.add(Dense(activation="relu", units=64))
start_cnn.add(Dense(activation="relu", units=32))
start_cnn.add(Dense(activation="softmax", units=1))

start_cnn.summary()

# Compiling the CNN

# start_cnn.compile(Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
batch_size=len(X_train)/234
batch_size=int(batch_size)
# start_cnn.fit(X_train, y_train, steps_per_epoch=234, epochs=100, validation_data=(X_test, y_test))
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_test,
    'y_val': y_test
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=start_cnn,optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',dataset=dataset,iters=100,batch_size=batch_size,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_test)
numpy.save(Y_test_name+'.npy', y_test)
