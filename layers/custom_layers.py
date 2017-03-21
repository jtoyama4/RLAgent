from keras.engine.topology import Layer
from keras import backend as K
import numpy as np

class DilatedGateLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DilatedGateLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Vx = self.add_weight(shape=(input_shape[1], self.output_dim),
                                  initializer='uniform',
                                  trainable=True)

        self.Vy = self.add_weight(shape=(input_shape[1], self.output_dim),
                                  initializer='uniform',
                                  trainable=True)

        super(DilatedGateLayer).build(input_shape)

    def call(self, t):
        x1, y1, z = t
        #xv = K.variable(np.random.random((32, self.z_dim)))
        x2 = K.dot(z, K.transpose(self.Vx))
        x = K.tanh(x1 + x2[:, None, :])

        #yv = K.variable(np.random.random((32, self.z_dim)))
        y2 = K.dot(z, K.transpose(self.Vy))
        y = K.sigmoid(y1 + y2[:, None, :])

        return x * y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)