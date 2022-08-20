import tensorflow as tf

class ResnetConv1D(tf.keras.Model):
  def __init__(self, kernel_size, filters):
    super(ResnetConv1D, self).__init__(name='')
    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv1D(filters1, 1)
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv1D(filters2, kernel_size, padding='causal')
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.conv2c = tf.keras.layers.Conv1D(filters3, 1)
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)


class ResnetDense(tf.keras.Model):
  def __init__(self, output_sizes):
    super(ResnetDense, self).__init__(name='')
    ops1, ops2, ops3 = output_sizes

    self.dense1 = tf.keras.layers.Dense(ops1)
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.dense2 = tf.keras.layers.Dense(ops2)
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.dense3 = tf.keras.layers.Dense(ops3)
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.dense1(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.dense2(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.dense3(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)
