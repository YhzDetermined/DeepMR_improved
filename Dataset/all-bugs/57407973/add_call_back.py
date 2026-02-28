import os

import numpy
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_classification(1000, 28)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(28, input_dim=28, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(200, kernel_initializer='normal', activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(300, kernel_initializer='normal', activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(300, kernel_initializer='normal', activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(150, kernel_initializer='normal', activation='sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# history = model.fit(X_train, y_train,
#                     epochs=34,
#                     batch_size=32,
#                     validation_data=(X_val, y_val),
#                     verbose=1)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_val,
    'y_val': y_val
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='binary_crossentropy',dataset=dataset,iters=34,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_val)
numpy.save(Y_test_name+'.npy', y_val)