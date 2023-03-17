import datetime
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.engine.base_layer import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, InputLayer, Add, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
import tensorflow.keras.backend as K

class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(x):
    return x*K.tanh(K.softplus(x))

get_custom_objects().update({'Mish': Mish(mish)})

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def callbacks(lr_factor, lr_patience, es_patience):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tc = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    lr = ReduceLROnPlateau(monitor = 'val_loss', factor = lr_factor, patience = lr_patience, verbose = 0)
    es = EarlyStopping(monitor = 'val_loss',patience = es_patience, verbose = 0, mode = 'min', restore_best_weights = True)
    return [lr,es,tc]

def optimizer(lr, amsgrad=True):

    return tfa.optimizers.AdamW(
        weight_decay=0.0000004, learning_rate=lr, beta_1=0.95, amsgrad=amsgrad
    )

def nn_model(X_train, activation=activation, lr=lr, dropout_pct=dropout_pct):

    activation = tfa.activations.mish

    inputs = Input(shape=X_train.shape[1:])

    f = Dense(64, activation=activation)(inputs)
    # xb = BatchNormalization()(f)

    fa1 = Dense(32, activation=activation)(f)
    fa2 = Dense(16, activation=activation)(fa1)
    fb1 = Dense(32, activation=activation)(f)
    fb2 = Dense(16, activation=activation)(fb1)
    fc1 = Dense(32, activation=activation)(f)
    fc2 = Dense(16, activation=activation)(fc1)
    fd1 = Dense(32, activation=activation)(f)
    fd2 = Dense(16, activation=activation)(fd1)
    fe1 = Dense(32, activation=activation)(f)
    fe2 = Dense(16, activation=activation)(fe1)
    f3 = Add()([fa2, fb2, fc2, fd2, fe2])
    xb = BatchNormalization()(f3)

    x = Dense(32, activation=activation)(xb)
    x = Dense(16, activation=activation)(x)
    add = Add()([xb, x])
    #x = Dropout(dropout_pct)(x)
    x = Dense(32, activation=activation)(add)
    x = Dense(16, activation=activation)(x)
    add = Add()([add, x])
    #x = Dropout(dropout_pct)(x)
    x = Dense(32, activation=activation)(add)
    x = Dense(16, activation=activation)(x)
    add = Add()([add, x])
    #x = Dropout(dropout_pct)(x)
    x = Dense(32, activation=activation)(add)
    x = Dense(16, activation=activation)(x)
    add = Add()([add, x])
    #x = Dropout(dropout_pct)(x)
    x = Dense(32, activation=activation)(add)
    x = Dense(16, activation=activation)(x)
    add = Add()([add, x])

    x = Dense(8, activation=activation)(add)

    output = Dense(1, activation='linear')(x)
    # output = Dense(y_train.shape[1:], activation='linear')(x)

    model = Model(inputs, output)

    # Compile the model
    model.compile(
    loss=rmse,
    optimizer=optimizer(lr),
    metrics=[rmse]
    )

    return model