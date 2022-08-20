import tensorflow as tf
import numpy as np

class Attention(tf.keras.layers.Layer):
    """
    Multi-Head Convolutional Self Attention Layer
    """

    def __init__(self, dk, dv, num_heads, filter_size):
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.num_heads = num_heads

        self.conv_q = tf.keras.layers.Conv1D(dk * num_heads, filter_size, padding='causal')
        self.conv_k = tf.keras.layers.Conv1D(dk * num_heads, filter_size, padding='causal')
        self.dense_v = tf.keras.layers.Dense(dv * num_heads)
        self.dense1 = tf.keras.layers.Dense(dv, activation='relu')
        self.dense2 = tf.keras.layers.Dense(dv)

    def split_heads(self, x, batch_size, dim):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size, time_steps, _ = tf.shape(inputs)

        q = self.conv_q(inputs)
        k = self.conv_k(inputs)
        v = self.dense_v(inputs)

        q = self.split_heads(q, batch_size, self.dk)
        k = self.split_heads(k, batch_size, self.dk)
        v = self.split_heads(v, batch_size, self.dv)

        mask = 1 - tf.linalg.band_part(tf.ones((batch_size, self.num_heads, time_steps, time_steps)), -1, 0)

        dk = tf.cast(self.dk, tf.float32)

        score = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk) + mask * -1e9)

        outputs = tf.matmul(score, v)

        outputs = tf.transpose(outputs, perm=[0, 2, 1, 3])
        outputs = tf.reshape(outputs, (batch_size, time_steps, -1))

        outputs = self.dense1(outputs)
        outputs = self.dense2(outputs)

        return outputs


class Transformer(tf.keras.models.Model):
    """
    Time Series Transformer Model
    """

    def __init__(self, dk, dv, num_heads, filter_size):
        super().__init__()
        # Note that multiple Attentions are used in the article. For simplicity, this demo uses only one layer
        self.attention = Attention(dk, dv, num_heads, filter_size)
        self.dense_mu = tf.keras.layers.Dense(1)
        self.dense_sigma = tf.keras.layers.Dense(1, activation='softplus')

    def call(self, inputs):
        outputs = self.attention(inputs)
        mu = self.dense_mu(outputs)
        sigma = self.dense_sigma(outputs)

        return [mu, sigma]