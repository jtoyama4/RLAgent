from keras.models import load_model
from keras import backend as K

def gated_activation(t):
    x1, y1, z, zz = t

    x = K.tanh(x1 + z[:, None, :])
    y = K.sigmoid(y1 + zz[:, None, :])

    return x * y

a = load_model('dynamics/generator.hdf5', custom_objects={"gated_activation": gated_activation})