import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input, InputLayer, Add, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def nn_model(X_train,y_train,X_val,y_val,X_test,lr):
    
    
    # Create a sequential model
    model= Sequential([
    tf.keras.layers.Input(shape = X_train.shape[1:]),
    Dense(1024, activation='swish'),
    BatchNormalization(),
    Dense(512, activation='swish'),
    BatchNormalization(),
    Dense(512, activation='swish'),
    BatchNormalization(),
    Dense(128, activation='swish'),
    BatchNormalization(),
    Dense(64, activation='swish'),
    BatchNormalization(),
    Dense(1,   activation = 'linear')
    ])
    
    # Compile the model
    model.compile(
    loss=rmse,
    optimizer=Adam(learning_rate = lr),
    metrics=[rmse]
    )
    return model

def callbacks(lr_factor, lr_patience, es_patience):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tc = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, verbose = 0)
    es = EarlyStopping(monitor = 'val_loss',patience = 12, verbose = 0, mode = 'min', restore_best_weights = True)
    return [lr,es,tc]


