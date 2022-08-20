from tstransformer import Transformer
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

t = np.arange(0, 1000, 0.1)

sin_t = np.sin(t).reshape(1, t.shape[0],1)
noisy_t = sin_t + 0.4 * np.random.random(t.shape)


model = Transformer(10, 10, 5, 2)
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer, loss = tf.keras.losses.MSE)
model.fit(noisy_t, noisy_t)

pred = model.predict(noisy_t)

plt.plot(noisy_t[0][:200][0])
plt.plot(noisy_t[:200])
plt.show()