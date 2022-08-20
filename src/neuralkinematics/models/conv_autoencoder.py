import tensorflow as tf
import numpy as np
class DenseAutoencoder(object):
    tf.random.set_seed(10)

    def __init__(self, input_streams, latent_features = 9, lr = 0.001, batch_size = 50, epochs = 500, verbose = True):
        self.encoder = None
        self.decoder = None
        self.n_epochs = epochs
        self.batch_size = batch_size
        self.streams = input_streams
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        self.latent_features = latent_features
        self.verbose = verbose
    def make_encoder(self):
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.Conv1D(100, 2, input_shape = [None, self.streams], padding = 'same'))
        self.encoder.add(tf.keras.layers.Conv1D(50, 2))
        self.encoder.add(tf.keras.layers.Dense(self.latent_features))

    def make_decoder(self):
        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.Dense(50, input_shape = [None, self.latent_features]))
        self.decoder.add(tf.keras.layers.Conv1DTranspose(100, 2))
        self.decoder.add(tf.keras.layers.Dense(self.streams))


    @tf.function
    def train_step(self, X_b):
        with tf.GradientTape() as enc_tape , tf.GradientTape() as dec_tape:
            latent = self.encoder(X_b)
            output = self.decoder(latent)

            loss = tf.keras.losses.MSE(X_b, output)

        enc_grad = enc_tape.gradient(loss, self.encoder.trainable_variables)
        dec_grad = dec_tape.gradient(loss, self.decoder.trainable_variables)

        self.optimizer.apply_gradients(zip(enc_grad, self.encoder.trainable_variables))
        self.optimizer.apply_gradients(zip(dec_grad, self.decoder.trainable_variables))
        return loss

    def fit(self, input_signal):
        self.make_decoder()
        self.make_encoder()

        batch_size = self.batch_size

        n_batches = input_signal.shape[0] // batch_size
        losses = []

        for i in range(self.n_epochs):
            loss = 0
            for b in range(n_batches):

                b_st = b * batch_size
                b_en = min((b + 1) * batch_size, input_signal.shape[0])
                i_b = input_signal[b_st:b_en]
                l = self.train_step(i_b.reshape(1, i_b.shape[0], i_b.shape[1]))
                if np.isnan(l).any() :
                    return losses
                loss += np.mean(l)
            losses.append(np.mean(loss))


            if self.verbose and not i % 20:
                print('training epoch {}: loss = {}'.format(i + 1, loss))

        return losses