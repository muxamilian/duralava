import sys
import tensorflow as tf
from glob import glob
import os
from tensorflow.keras import layers
from datetime import datetime
import argparse
import random
from tensorflow.python.ops.gen_batch_ops import batch
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser(description="Either train a model, evaluate an existing one on a dataset or run live.")
parser.add_argument('--data_dir', type=str, default='frames', help='Directory with training data.')
parser.add_argument('--weights', type=str, default='', help='Directory with training data.')

args = parser.parse_args()

# Disable GPU
# tf.config.experimental.set_visible_devices([], 'GPU')

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)

# lr = 1e-4
# lr = 1e-2
lr = 2e-4
img_size = 32
batch_size = 64
n_items = 100000
epochs = sys.maxsize
noise_dim = 128
num_examples_to_generate = 16
regularization_multiplier = 0.1
reset_probability = 1.0
every_nth = 1
seq_len = 1

def process_img(file_path, img_size):
  img = tf.io.read_file(file_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, size=(img_size, img_size), method='area')
  img = tf.image.convert_image_dtype(img, tf.uint8)
  return img.numpy()

# def make_generator_model():
#   model = tf.keras.Sequential()

#   model.add(layers.Dense(256))
#   model.add(layers.LeakyReLU(alpha=0.1))
#   model.add(layers.Dense(256))
#   model.add(layers.LeakyReLU(alpha=0.1))
#   model.add(layers.Reshape((1, 1, 256)))
#   model.add(layers.Conv2DTranspose(256, kernel_size=4, strides=4, padding='same'))
#   model.add(layers.LeakyReLU(alpha=0.1))
#   model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
#   model.add(layers.LeakyReLU(alpha=0.1))
#   model.add(layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
#   model.add(layers.LeakyReLU(alpha=0.1))
#   model.add(layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'))
#   model.add(layers.LeakyReLU(alpha=0.1))
#   model.add(layers.Conv2DTranspose(3, kernel_size=4, strides=2, activation=None, padding='same'))

#   recurrent_part = tf.keras.Sequential()
#   recurrent_part.add(layers.Dense(256))
#   recurrent_part.add(layers.LeakyReLU(alpha=0.1))
#   recurrent_part.add(layers.Dense(256))
#   recurrent_part.add(layers.LeakyReLU(alpha=0.1))
#   recurrent_part.add(layers.Dense(256))
#   recurrent_part.add(layers.LeakyReLU(alpha=0.1))
#   recurrent_part.add(layers.Dense(noise_dim))

#   return model, recurrent_part

def make_generator_model():
  return tf.keras.Sequential(
      [
        tf.keras.Input(shape=(noise_dim,)),
        # layers.Dense(512),
        # layers.LeakyReLU(alpha=0.1),
        layers.Reshape ((1, 1, noise_dim)),
        layers.Conv2DTranspose(256, kernel_size=4, strides=4, padding='same', use_bias=False),
        layers.BatchNormalization(),
        # layers.LeakyReLU(alpha=0.1),
        layers.ReLU(),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        # layers.LeakyReLU(alpha=0.1),
        layers.ReLU(),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        # layers.LeakyReLU(alpha=0.1),
        layers.ReLU(),
        layers.Conv2DTranspose(3, kernel_size=4, strides=2, activation='sigmoid', padding='same')
      ],
      name="generator",
  )

# def make_discriminator_model():
#   model = tf.keras.Sequential()

#   model.add(layers.Input(shape=(img_size, img_size, 3)))
#   model.add(layers.Conv2D(32, (4, 4), padding='same', strides=2))
#   model.add(layers.LeakyReLU(alpha=0.1))
#   model.add(layers.Conv2D(64, (4, 4), padding='same', strides=2))
#   model.add(layers.LeakyReLU(alpha=0.1))
#   model.add(layers.Conv2D(128, (4, 4), padding='same', strides=2))
#   model.add(layers.LeakyReLU(alpha=0.1))
#   model.add(layers.Conv2D(256, (4, 4), padding='same', strides=2))
#   model.add(layers.LeakyReLU(alpha=0.1))
#   model.add(layers.Conv2D(256, (4, 4), padding='same', strides=4))
#   model.add(layers.LeakyReLU(alpha=0.1))
#   model.add(layers.Flatten())
#   model.add(layers.Dense(256))
#   model.add(layers.LeakyReLU(alpha=0.1))
#   model.add(layers.Dense(noise_dim))

#   recurrent_part = tf.keras.Sequential()
#   recurrent_part.add(layers.Dense(256))
#   recurrent_part.add(layers.LeakyReLU(alpha=0.1))
#   recurrent_part.add(layers.Dense(256))
#   recurrent_part.add(layers.LeakyReLU(alpha=0.1))
#   recurrent_part.add(layers.Dense(256))
#   recurrent_part.add(layers.LeakyReLU(alpha=0.1))
#   recurrent_part.add(layers.Dense(noise_dim))

#   end_part = tf.keras.Sequential()
#   end_part.add(layers.Dense(256))
#   end_part.add(layers.LeakyReLU(alpha=0.1))
#   end_part.add(layers.Dense(1))

#   return model, recurrent_part, end_part

def make_discriminator_model():
  return tf.keras.Sequential(
    [
      layers.Input(shape=(img_size, img_size, 3)),
      layers.Conv2D(64, (4, 4), padding='same', strides=2),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2D(128, (4, 4), padding='same', strides=2, use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2D(256, (4, 4), padding='same', strides=2, use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2D(128, (4, 4), padding='same', strides=4),
      # layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Flatten(),
      # layers.Dense(512),
      # layers.LeakyReLU(alpha=0.1),
      # layers.Dropout(0.2),
      layers.Dense(1, activation=None),
    ],
    name="discriminator",
)

x_files = sorted(glob(f'{args.data_dir}/*.jpg'))[:n_items]
cache_file_name = f'{args.data_dir}.pickle'
if os.path.isfile(cache_file_name):
  with open(cache_file_name, 'rb') as f:
    all_images = pickle.load(f)
else:
  all_images = []
  for item in tqdm(x_files):
    all_images.append(process_img(item, img_size))
  with open(cache_file_name, 'wb') as f:
    pickle.dump(all_images, f)

batches_per_epoch = int((len(x_files)-every_nth*seq_len)/(batch_size*seq_len))

def data_generator():
  while True:
    indices = list(range(len(all_images)-every_nth*seq_len))
    random.shuffle(indices)

    for i in range(batches_per_epoch):
      current_batch_indices = indices[i*batch_size:(i+1)*batch_size]
      batch = [[all_images[index+every_nth*j] for index in current_batch_indices] for j in range(seq_len)]
      yield (batch,)

class CustomModel(tf.keras.Model):

  def __init__(self):
    super(CustomModel, self).__init__()

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    self.gen_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    self.disc_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)

    self.seed = tf.random.normal([num_examples_to_generate, noise_dim])
    self.val_data = self.transform_images(next(data_generator()))[0]

    self.gen_loss_tracker = tf.keras.metrics.Mean(name="gen_loss")
    self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")
    self.gen_regularization_loss_tracker = tf.keras.metrics.Mean(name="gen_regularization_loss")
    self.gen_mean_tracker = tf.keras.metrics.Mean(name="gen_mean")
    self.gen_std_tracker = tf.keras.metrics.Mean(name="gen_std")
    self.gen_skew_tracker = tf.keras.metrics.Mean(name="gen_skew")
    self.gen_kurt_tracker = tf.keras.metrics.Mean(name="gen_kurt")
    # self.gen_std_mean_tracker = tf.keras.metrics.Mean(name="gen_std_mean")

    # This method returns a helper function to compute cross entropy loss
    self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # self.noise = tf.Variable(tf.random.normal([batch_size, noise_dim]))
    self.noise = tf.random.normal([batch_size, noise_dim])

    # self.generator, self.recurrent_generator = make_generator_model()
    # self.discriminator, self.recurrent_discriminator, self.discriminator_end = make_discriminator_model()
    self.generator = make_generator_model()
    self.generator.summary()
    self.discriminator = make_discriminator_model()
    self.discriminator.summary()

  @tf.function
  def discriminator_loss(self, real_output, fake_output):
      real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
      fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
      total_loss = real_loss + fake_loss
      return total_loss

  @tf.function
  def generator_loss(self, fake_output):
      return self.cross_entropy(tf.ones_like(fake_output), fake_output)

  @tf.function
  def call(self, _, training=False):
    return self.inference()

  @tf.function
  def transform_images(self, images):
    out_images = tf.stack([tf.stack([tf.image.convert_image_dtype(tf.convert_to_tensor(item, dtype=tf.uint8), tf.float32) for item in sublist]) for sublist in images])
    return out_images

  @tf.function
  def inference(self):
    noise = self.seed
    
    all_noises = []
    all_generated_images = []
    for _ in range(seq_len):
      generated_images = self.generator(noise, training=False)
      all_generated_images.append(generated_images)
      # noise = self.recurrent_generator(noise, training=False)
      # all_noises.append(noise)
    return all_generated_images, all_noises

  @tf.function
  def train_step(self, images):
    images = self.transform_images(images[0])
    self.noise = tf.random.normal([batch_size, noise_dim])
    current_noise = self.noise

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      # prev_real_output  = tf.zeros((batch_size, noise_dim))
      # prev_fake_output = tf.zeros((batch_size, noise_dim))
      # gen_losses = []
      # gen_outputs = []
      # disc_losses = []

      for i in range(seq_len):
        generated_images = self.generator(current_noise, training=True)
        # current_noise = self.recurrent_generator(current_noise, training=True)
        # gen_outputs.append(current_noise)

        # real_first_output = tf.reshape(self.discriminator(images[i,...], training=True), (batch_size, noise_dim))
        # concat_real = tf.concat((real_first_output, prev_real_output), axis=-1)
        # prev_real_output = self.recurrent_discriminator(concat_real, training=True)
        # real_output = self.discriminator_end(prev_real_output, training=True)
        real_output = self.discriminator(images[i,...], training=True)

        # fake_first_output = tf.reshape(self.discriminator(generated_images, training=True), (batch_size, noise_dim))
        # concat_fake = tf.concat((fake_first_output, prev_fake_output), axis=-1)
        # prev_fake_output = self.recurrent_discriminator(concat_fake, training=True)
        # fake_output = self.discriminator_end(prev_fake_output, training=True)
        fake_output = self.discriminator(generated_images, training=True)

        # gen_losses.append(self.generator_loss(fake_output))
        # disc_losses.append(self.discriminator_loss(real_output, fake_output))

        gen_loss = self.generator_loss(fake_output)
        disc_loss = self.discriminator_loss(real_output, fake_output)

      # gen_loss = tf.reduce_mean(tf.stack(gen_losses))
      # gen_all_outputs = tf.concat(gen_outputs, axis=0)
      # gen_mean = tf.reduce_mean(gen_all_outputs)
      # gen_std = tf.math.reduce_std(gen_all_outputs)
      # gen_skew = tf.reduce_mean((gen_all_outputs - gen_mean)**3)/gen_std**3
      # gen_kurt = tf.reduce_mean((gen_all_outputs - gen_mean)**4)/gen_std**4
      # gen_std_mean = tf.math.reduce_std(tf.reduce_mean(gen_all_outputs, axis=(0,2)))
      # gen_regularization_loss = \
      #     gen_mean**2 + \
      #     (gen_std - 1)**2 + \
      #     gen_skew**2 + \
      #     (gen_kurt - 3)**2# + \

      # disc_loss = tf.reduce_mean(tf.stack(disc_losses))

      # gradients_of_generator = gen_tape.gradient(gen_loss + regularization_multiplier*gen_regularization_loss, (
      gradients_of_generator = gen_tape.gradient(gen_loss, 
        self.generator.trainable_variables)# + 
        # self.recurrent_generator.trainable_variables))
      gradients_of_discriminator = disc_tape.gradient(disc_loss, 
        self.discriminator.trainable_variables)# + 
        # self.recurrent_discriminator.trainable_variables + 
        # self.discriminator_end.trainable_variables))

      self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))# + 
        # self.recurrent_generator.trainable_variables)))
      self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))# + 
        # self.recurrent_discriminator.trainable_variables + 
        # self.discriminator_end.trainable_variables)))

      self.gen_loss_tracker.update_state(gen_loss)
      self.disc_loss_tracker.update_state(disc_loss)
      # self.gen_regularization_loss_tracker.update_state(gen_regularization_loss)
      # self.gen_mean_tracker.update_state(gen_mean)
      # self.gen_std_tracker.update_state(gen_std)
      # self.gen_skew_tracker.update_state(gen_skew)
      # self.gen_kurt_tracker.update_state(gen_kurt)

      return {
        "gen_loss": self.gen_loss_tracker.result(), "disc_loss": self.disc_loss_tracker.result()}#,
        # "gen_regularization_loss": self.gen_regularization_loss_tracker.result(),
        # "gen_mean": self.gen_mean_tracker.result(),
        # "gen_std": self.gen_std_tracker.result(),
        # "gen_skew": self.gen_skew_tracker.result(),
        # "gen_kurt": self.gen_kurt_tracker.result()}

class CustomCallback(tf.keras.callbacks.Callback):

  def on_epoch_begin(self, epoch, logs):
    if epoch == 0:
      with file_writer.as_default():
        tf.summary.image("In imgs first", self.model.val_data[0,...], step=epoch, max_outputs=num_examples_to_generate)
        # tf.summary.image("In imgs second", self.model.val_data[1,...], step=epoch, max_outputs=num_examples_to_generate)
        # tf.summary.image("In imgs third", self.model.val_data[2,...], step=epoch, max_outputs=num_examples_to_generate)
        # tf.summary.image("In imgs middle", self.model.val_data[int(self.model.val_data.shape[0]/2),...], step=epoch, max_outputs=num_examples_to_generate)
        # tf.summary.image("In imgs end", self.model.val_data[-1,...], step=epoch, max_outputs=num_examples_to_generate)
    self.model.gen_loss_tracker.reset_states()
    self.model.disc_loss_tracker.reset_states()
    # self.model.gen_regularization_loss_tracker.reset_states()
    # self.model.gen_mean_tracker.reset_states()
    # self.model.gen_std_tracker.reset_states()
    # self.model.gen_skew_tracker.reset_states()
    # self.model.gen_kurt_tracker.reset_states()

  def on_epoch_end(self, epoch, logs=None):
    all_generated_images, all_noises = self.model.inference()

    first = all_generated_images[0]
    # first_noise = all_noises[0]     

    # second = all_generated_images[1]
    # second_noise = all_noises[1]          

    # third = all_generated_images[2]
    # third_noise = all_noises[2]          

    # middle = all_generated_images[int(len(all_generated_images)/2)]
    # middle_noise = all_noises[int(len(all_noises)/2)]          

    # end = all_generated_images[-1]
    # end_noise = all_noises[-1]          

    with file_writer.as_default():
      tf.summary.image("Out imgs first", first, step=epoch, max_outputs=num_examples_to_generate)
      # tf.summary.image("Out imgs second", second, step=epoch, max_outputs=num_examples_to_generate)
      # tf.summary.image("Out imgs third", third, step=epoch, max_outputs=num_examples_to_generate)
      # tf.summary.image("Out imgs middle", middle, step=epoch, max_outputs=num_examples_to_generate)
      # tf.summary.image("Out imgs end", end, step=epoch, max_outputs=num_examples_to_generate)

      # tf.summary.histogram("Out noise first", tf.reshape(first_noise, (-1,)), step=epoch)
      # tf.summary.histogram("Out noise second", tf.reshape(second_noise, (-1,)), step=epoch)
      # tf.summary.histogram("Out noise third", tf.reshape(third_noise, (-1,)), step=epoch)
      # tf.summary.histogram("Out noise middle", tf.reshape(middle_noise, (-1,)), step=epoch)
      # tf.summary.histogram("Out noise end", tf.reshape(end_noise, (-1,)), step=epoch)

      for key in logs:
        tf.summary.scalar(key, logs[key], step=epoch)

model = CustomModel()
model.compile()#, run_eagerly=True)
if args.weights != '':
  print("Loading weights from", args.weights)
  model.load_weights(args.weights)
model.fit(x=data_generator(),
          epochs=epochs,
          callbacks=[
            CustomCallback(), 
            tf.keras.callbacks.ModelCheckpoint(
             	os.path.join(logdir, "weights.{epoch:02d}"), verbose=1, save_weights_only=True, save_freq=10*batches_per_epoch)
          ],
          shuffle=False,
          steps_per_epoch=batches_per_epoch)
          # steps_per_epoch=1)

