import tensorflow as tf
import numpy as np
class MVTSGAN(object):
    tf.random.set_seed(10)
    def __init__(self, input_streams, output_streams = None, lr = 0.001, l2_gen = 0.0001, saturation = True, epochs = 100, batch_len = 100, rand_noise = False, verbose = False, lr_reduce=True):
        self.input_streams = input_streams
        self.output_streams = output_streams if not output_streams == None else input_streams
        self.generator = None
        self.discriminator = None
        self.lr = lr
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)
        self.l2_gen = l2_gen
        self.saturate = saturation
        self.n_epochs = epochs
        self.batch_size = batch_len
        self.rand_noise = rand_noise
        self.verbose = verbose
        self.min_lr = self.lr * 0.005

    def make_aligner(self):

        self.aligner = tf.keras.Sequential()

        self.aligner.add(tf.keras.layers.Dense(4, input_shape = (None, self.input_streams)))
        self.aligner.add(tf.keras.layers.Dense(10))
        self.aligner.add(tf.keras.layers.Dense(self.input_streams))

    def make_generator(self):

        self.generator = tf.keras.Sequential()

        self.generator.add(tf.keras.layers.Conv1D(100, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l1_l2(l2=self.l2_gen), input_shape=(None, self.input_streams)))
        self.generator.add(tf.keras.layers.Conv1D(100, 3, padding='same', activation='tanh'))
        self.generator.add(tf.keras.layers.Conv1D(100, 2, padding='same'))
        #
        # self.generator.add(tf.keras.layers.Conv1D(100, 2, padding='same'))

        # self.generator.add(tf.keras.layers.Dense(20, activation='linear'))
        # self.generator.add(tf.keras.layers.Dense(20, activation='linear'))
        # self.generator.add(tf.keras.layers.Dense(20, activation='linear'))
        # self.generator.add(tf.keras.layers.Dense(100, activation='linear'))

        self.generator.add(tf.keras.layers.Dense(100))

        self.generator.add(tf.keras.layers.Dense(self.output_streams))

    def make_discriminator(self):
        self.discriminator = tf.keras.Sequential()

        self.discriminator.add(tf.keras.layers.Conv1D(100, 3, padding='same', input_shape=(None, self.output_streams)))
        # self.discriminator.add(tf.keras.layers.Conv1D(100, 3, padding='same'))
        self.discriminator.add(tf.keras.layers.MaxPool1D(2))
        self.discriminator.add(tf.keras.layers.Conv1D(100, 3, padding='same', activation='tanh'))
        self.discriminator.add(tf.keras.layers.MaxPool1D(4))

        # self.discriminator.add(tf.keras.layers.Conv1D(100, 3, padding='same', activation='tanh'))
        # self.discriminator.add(tf.keras.layers.MaxPool1D(8))

        self.discriminator.add(tf.keras.layers.Dense(50, activation='tanh'))
        # self.discriminator.add(tf.keras.layers.Dense(20, activation='linear'))
        # self.discriminator.add(tf.keras.layers.Dense(10, activation='linear'))
        self.discriminator.add(tf.keras.layers.Dense(50, activation='linear'))

        self.discriminator.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    @tf.function
    def train_step(self, real_input, real_target):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as ali_tape:

            if self.rand_noise:
                real_input += tf.random.normal(real_input.shape, stddev=1.)

            reconstructed = self.generator(self.aligner(real_input))
            fake_out = (self.discriminator(reconstructed))
            real_out = (self.discriminator(real_target))

            discriminator_loss = -tf.reduce_mean(tf.math.log(real_out) + tf.math.log(1 - fake_out))

            if self.saturate:
                generator_loss = tf.reduce_mean(1 - tf.math.log(fake_out)) + self.generator.losses
            else:
                generator_loss = -tf.reduce_mean(tf.math.log(fake_out))

        gen_gradient = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
        dis_gradient = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        ali_gradient = ali_tape.gradient(generator_loss,self.aligner.trainable_variables )

        self.optimizer.apply_gradients(zip(ali_gradient, self.aligner.trainable_variables))
        self.optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        self.optimizer.apply_gradients(zip(dis_gradient, self.discriminator.trainable_variables))

        return generator_loss, discriminator_loss

    def fit(self, input_signal, target_signal):
        self.make_aligner()
        self.make_generator()
        self.make_discriminator()

        batch_size = self.batch_size

        n_batches = input_signal.shape[0] // batch_size
        g_losses = []
        d_losses = []
        for i in range(self.n_epochs):
            gen_loss = 0
            dis_loss = 0
            for b in range(n_batches):

                b_st = b * batch_size
                b_en = min((b+1) * batch_size, input_signal.shape[0])

                g, d = self.train_step(input_signal[b_st:b_en].reshape(1, b_en - b_st, self.input_streams), target_signal[b_st:b_en].reshape(1, b_en - b_st, self.output_streams))
                if np.isnan(g).any() or np.isnan(d).any():
                    return g_losses, d_losses
                gen_loss += g
                dis_loss += d
            g_losses.append(np.mean(gen_loss))
            d_losses.append(dis_loss)
            # if not i%20:
            #     self.lr = max(self.min_lr, self.lr*0.8)

            if self.verbose and not i %20:
                print('training epoch {}: g loss = {}: d loss = {}:'.format(i+1, gen_loss, dis_loss))

        return g_losses, d_losses