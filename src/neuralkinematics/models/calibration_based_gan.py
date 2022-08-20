import tensorflow as tf
import numpy as np

class MVTSGAN(object):

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

    def make_calibrator(self):

        self.calibrator = tf.keras.Sequential()

        self.calibrator.add(tf.keras.layers.Dense(2, input_shape=[None,6]))
        self.calibrator.add(tf.keras.layers.Dense(100))

    def make_extractor(self):

        self.extractor = tf.keras.Sequential()

        self.extractor.add(tf.keras.layers.Conv1D(50, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l1_l2(l2=self.l2_gen), input_shape=(None, self.input_streams)))
        self.extractor.add(tf.keras.layers.Conv1D(50, 3, padding='same', activation='tanh'))
        self.extractor.add(tf.keras.layers.Conv1D(50, 3, padding='same'))

        self.extractor.add(tf.keras.layers.Dense(100, activation='linear'))

    def make_generator(self):
        self.generator = tf.keras.Sequential()

        self.generator.add(tf.keras.layers.Dense(50, activation='linear', input_shape= [None, 200]))

        self.generator.add(tf.keras.layers.Dense(4))

        self.generator.add(tf.keras.layers.Dense(self.output_streams))

    def make_discriminator(self):
        self.discriminator = tf.keras.Sequential()

        self.discriminator.add(tf.keras.layers.Conv1D(100, 3, padding='same', input_shape=(None, self.output_streams)))
        self.discriminator.add(tf.keras.layers.Conv1D(100, 3, padding='same'))
        self.discriminator.add(tf.keras.layers.Conv1D(50, 3, padding='same', activation='tanh'))
        self.discriminator.add(tf.keras.layers.Conv1D(25, 3, padding='same', activation='tanh'))
        self.discriminator.add(tf.keras.layers.Dense(50, activation='linear'))
        self.discriminator.add(tf.keras.layers.Dense(20, activation='linear'))
        self.discriminator.add(tf.keras.layers.Dense(10, activation='linear'))
        self.discriminator.add(tf.keras.layers.Dense(5, activation='linear'))

        self.discriminator.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    @tf.function
    def train_step(self, real_input, real_target, calibration):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as ext_tape, tf.GradientTape() as cal_tape:

            if self.rand_noise:
                real_input += tf.random.normal(real_input.shape, stddev=1.)
            extracted = self.extractor(real_input)
            calibrated = self.calibrator(calibration)
            concatenated = tf.keras.layers.concatenate([tf.reshape(extracted, (extracted.shape[1], extracted.shape[2])), calibrated])

            reconstructed = self.generator(tf.reshape(concatenated, [1, concatenated.shape[0], concatenated.shape[1]]))
            fake_out = (self.discriminator(reconstructed))
            real_out = (self.discriminator(real_target))

            discriminator_loss = -tf.reduce_mean(tf.math.log(real_out) + tf.math.log(1 - fake_out))

            if self.saturate:
                generator_loss = tf.reduce_mean(1 - tf.math.log(fake_out))
            else:
                generator_loss = -tf.reduce_mean(tf.math.log(fake_out))

        gen_gradient = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
        dis_gradient = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        cal_gradient = cal_tape.gradient(generator_loss, self.calibrator.trainable_variables)
        ext_gradient = ext_tape.gradient(generator_loss, self.extractor.trainable_variables)

        self.optimizer.apply_gradients(zip(cal_gradient, self.calibrator.trainable_variables))
        self.optimizer.apply_gradients(zip(ext_gradient, self.extractor.trainable_variables))
        self.optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        self.optimizer.apply_gradients(zip(dis_gradient, self.discriminator.trainable_variables))

        return generator_loss, discriminator_loss



    def predict(self, input_signal, calibration):
        extracted = self.extractor.predict(input_signal)
        calibrated = self.calibrator.predict(calibration)


        concatenated = np.append(extracted.reshape( (extracted.shape[1], extracted.shape[2])), calibrated, axis=1)

        reconstructed = self.generator.predict(tf.reshape(concatenated, [1, concatenated.shape[0], concatenated.shape[1]]))
        return reconstructed.reshape([input_signal.shape[1], self.output_streams])

    def fit(self, input_signal, target_signal, calibration):
        self.make_extractor()
        self.make_calibrator()
        self.make_generator()
        self.make_discriminator()

        batch_size = self.batch_size

        n_batches = input_signal.shape[0] // batch_size

        for i in range(self.n_epochs):
            gen_loss = 0
            dis_loss = 0
            for b in range(n_batches):

                b_st = b * batch_size
                b_en = min((b+1) * batch_size, input_signal.shape[0])

                g, d = self.train_step(input_signal[b_st:b_en].reshape(1, b_en - b_st, self.input_streams), target_signal[b_st:b_en].reshape(1, b_en - b_st, self.output_streams), calibration[b_st:b_en])

                gen_loss += g
                dis_loss += d

            # if not i%20:
            #     self.lr = max(self.min_lr, self.lr*0.8)

            if self.verbose and not i %20:
                print('training epoch {}: g loss = {}: d loss = {}:'.format(i+1, gen_loss, dis_loss))