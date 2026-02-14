import os

import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train
import time
start_time=time.time()
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
X, y = make_classification(100, 560 * 560, random_state=42)
X = X.reshape(-1, 560, 560)
y = to_categorical(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential(
    [
        Conv1D(320, 8, input_shape=(560, 560), activation="relu"),
        Dense(1500, activation="relu"),
        Dropout(0.6),
        Dense(750, activation="relu"),
        Dropout(0.6),
        GlobalMaxPooling1D(keepdims=True),
        Dense(1, activation='softmax')
    ]
)

#model.compile(optimizer=Adam(learning_rate=0.00001), loss="binary_crossentropy", metrics=['accuracy'])
#model1 = model.fit(X_train, y_train, batch_size=150, epochs=5, shuffle=True, verbose=1, validation_data=(X_val, y_val))
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_val,
    'y_val': y_val
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=Adam(learning_rate=0.00001),loss='binary_crossentropy',dataset=dataset,iters=5,batch_size=150,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_val)
numpy.save(Y_test_name+'.npy', y_val)
end_time=time.time()
print(end_time-start_time)