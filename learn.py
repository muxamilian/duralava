import sys
import tensorflow as tf
from glob import glob
import os
from tensorflow.keras import layers
from datetime import datetime
import argparse
import random
from tqdm import tqdm
import pickle
import numpy as np
from PIL import Image
import subprocess
import cv2
import time

parser = argparse.ArgumentParser(description="Either train a model, evaluate an existing one on a dataset or run live.")
parser.add_argument('--data_dir', type=str, default='frames', help='Directory with training data.')
parser.add_argument('--weights', type=str, default='', help='Path to weights, for example "logs/20220104-213105/weights.160".')
parser.add_argument('--mode', type=str, default='train', help='Can be "train", "live" or "video".')

args = parser.parse_args()

# Initial learning rate
lr = 1e-3
img_size = 64
batch_size = 32
# How many frames to take from the dataset for training. By default, take all
n_items = sys.maxsize
epochs = 200
# Dimension of the noise vector of the generator
noise_dim = 64
# Dimension of the internal state of the discriminator
disc_dim = 128
# Dimension of the internal state of the recurrent part of the discriminator
disc_recurrent_dim = 128
# How many examples to generate for visualization during training
num_examples_to_generate = 16
# The higher, the more the training aims to produces a noise vector in the generator which is shaped like a normal distribution
regularization_multiplier = 0.1
# After every training sequence, the internal state of the generator is reset with this probability. 
# A higher number makes training more stable (state is more often reset) but can lead to the generator making worse output, 
# thus not creating a credible lava lamp
reset_probability = 0.5
# FPS of the lava lamp training data
original_lava_lamp_fps = 30
# Letting the GAN learn of 30 FPS is too much. Thus only take every 6th frame (actual FPS is 5 then)
every_nth = 6
# Sequence length of the recurrent GAN. Higher is better but also more unstable. Higher seq_len needs more memory. 
seq_len = 20
# Output fps in 'live' or 'video' mode. If original_lava_lamp_fps==30 and every_nth==6 the actual FPS during training was 30/6==5.
# If output_fps is more than the FPS of during training, missing frames are linearly interpolated. 
output_fps = 20
# Length of the output video/apng to generate in seconds
evaluation_duration = 30

def process_img(file_path, img_size):
  img = tf.io.read_file(file_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, size=(img_size, img_size), method='area')
  img = tf.image.convert_image_dtype(img, tf.uint8)
  return img

def make_generator_model():
  gen = tf.keras.Sequential(
    [
      tf.keras.Input(shape=(noise_dim,)),
      layers.Reshape ((1, 1, noise_dim)),
      layers.Conv2DTranspose(256, kernel_size=4, strides=4, padding='same', use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2DTranspose(3, kernel_size=4, strides=2, activation='sigmoid', padding='same')
    ],
    name="generator",
  )

  recurrent_gen = tf.keras.Sequential(
    [
      layers.Input(shape=(noise_dim,)),
      layers.Dense(512),
      layers.LeakyReLU(alpha=0.2),
      layers.Dense(512),
      layers.LeakyReLU(alpha=0.2),
      layers.Dense(noise_dim)
    ],
    name="generator_recurrent"
  )

  return gen, recurrent_gen

def make_discriminator_model():
  disc = tf.keras.Sequential(
    [
      layers.Input(shape=(img_size, img_size, 3)),
      layers.Conv2D(32, (4, 4), padding='same', strides=2),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2D(64, (4, 4), padding='same', strides=2, use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2D(128, (4, 4), padding='same', strides=2, use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2D(256, (4, 4), padding='same', strides=2, use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2D(disc_dim, (4, 4), padding='same', strides=4),
      layers.Flatten(),
    ],
    name="discriminator",
  )

  recurrent_disc = tf.keras.Sequential(
    [
      layers.Input(shape=(disc_dim+disc_recurrent_dim,)),
      layers.Dense(512),
      layers.LeakyReLU(alpha=0.2),
      layers.Dense(512),
      layers.LeakyReLU(alpha=0.2),
      layers.Dense(disc_recurrent_dim),
    ],
    name="discriminator_recurrent"
  )

  end_disc = tf.keras.Sequential(
    [
      layers.Input(shape=(disc_recurrent_dim,)),
      layers.Dense(512),
      layers.LeakyReLU(alpha=0.2),
      layers.Dense(1, activation=None)
    ],
    name="end_discriminator"
  )

  return disc, recurrent_disc, end_disc

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

    self.seed = tf.random.normal([num_examples_to_generate, noise_dim])

    self.gen_loss_tracker = tf.keras.metrics.Mean(name="gen_loss")
    self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")
    self.gen_regularization_loss_tracker = tf.keras.metrics.Mean(name="gen_regularization_loss")
    self.gen_mean_tracker = tf.keras.metrics.Mean(name="gen_mean")
    self.gen_std_tracker = tf.keras.metrics.Mean(name="gen_std")
    self.gen_skew_tracker = tf.keras.metrics.Mean(name="gen_skew")
    self.gen_kurt_tracker = tf.keras.metrics.Mean(name="gen_kurt")

    self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    self.noise = tf.Variable(tf.random.normal([batch_size, noise_dim]))

    self.generator, self.recurrent_generator = make_generator_model()
    self.generator.summary()
    self.recurrent_generator.summary()
    self.discriminator, self.recurrent_discriminator, self.end_discriminator = make_discriminator_model()
    self.discriminator.summary()
    self.recurrent_discriminator.summary()
    self.end_discriminator.summary()

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

  def generate(self, noise):
    img = self.generator(noise[0:1,...], training=False)
    new_noise = self.recurrent_generator(noise, training=False)
    return img, new_noise

  @tf.function
  def transform_images(self, images):
    out_images = tf.stack([tf.stack([tf.image.convert_image_dtype(item, tf.float32) for item in sublist]) for sublist in images])
    return out_images

  @tf.function
  def inference(self):
    noise = self.seed
    
    all_noises = []
    all_generated_images = []
    for _ in range(seq_len):
      generated_images = self.generator(noise, training=False)
      all_generated_images.append(generated_images)
      noise = self.recurrent_generator(noise, training=False)
      gen_mean = tf.reduce_mean(noise)
      gen_std = tf.math.reduce_std(noise)
      noise = (noise - gen_mean)/gen_std
      all_noises.append(noise)
    return all_generated_images, all_noises

  @tf.function
  def train_step(self, images):
    images = self.transform_images(images[0])
    current_noise = self.noise

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      prev_real_output  = tf.zeros((batch_size, disc_recurrent_dim))
      prev_fake_output = tf.zeros((batch_size, disc_recurrent_dim))
      gen_losses = []
      gen_regularization_losses = []
      gen_means = []
      gen_stds = []
      gen_skews = []
      gen_kurts = []
      disc_losses = []

      for i in range(seq_len):
        generated_images = self.generator(current_noise, training=True)
        current_noise = self.recurrent_generator(current_noise, training=True)
        gen_mean = tf.reduce_mean(current_noise)
        gen_means.append(gen_mean)
        gen_std = tf.math.reduce_std(current_noise)
        gen_stds.append(gen_std)
        gen_skew = tf.reduce_mean((current_noise - gen_mean)**3)/gen_std**3
        gen_skews.append(gen_skew)
        gen_kurt = tf.reduce_mean((current_noise - gen_mean)**4)/gen_std**4
        gen_kurts.append(gen_kurt)
        gen_regularization_loss = \
            gen_mean**2 + \
            (gen_std - 1)**2 + \
            gen_skew**2 + \
            (gen_kurt - 3)**2
        gen_regularization_losses.append(gen_regularization_loss)
        current_noise = (current_noise - gen_mean)/gen_std

        real_first_output = tf.reshape(self.discriminator(images[i,...], training=True), (batch_size, disc_dim))
        concat_real = tf.concat((real_first_output, prev_real_output), axis=-1)
        prev_real_output = self.recurrent_discriminator(concat_real, training=True)
        real_output = self.end_discriminator(prev_real_output, training=True)

        fake_first_output = tf.reshape(self.discriminator(generated_images, training=True), (batch_size, disc_dim))
        concat_fake = tf.concat((fake_first_output, prev_fake_output), axis=-1)
        prev_fake_output = self.recurrent_discriminator(concat_fake, training=True)
        fake_output = self.end_discriminator(prev_fake_output, training=True)

        gen_losses.append(self.generator_loss(fake_output))
        disc_losses.append(self.discriminator_loss(real_output, fake_output))

      gen_loss = tf.reduce_mean(tf.stack(gen_losses))
      gen_regularization_loss = tf.reduce_mean(tf.stack(gen_regularization_losses))

      disc_loss = tf.reduce_mean(tf.stack(disc_losses))

      gradients_of_generator = gen_tape.gradient(gen_loss + regularization_multiplier*gen_regularization_loss, 
        self.generator.trainable_variables + 
        self.recurrent_generator.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, 
        self.discriminator.trainable_variables + 
        self.recurrent_discriminator.trainable_variables + 
        self.end_discriminator.trainable_variables)

      self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables + 
        self.recurrent_generator.trainable_variables))

      self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables + 
        self.recurrent_discriminator.trainable_variables + 
        self.end_discriminator.trainable_variables))

      self.gen_loss_tracker.update_state(gen_loss)
      self.disc_loss_tracker.update_state(disc_loss)
      self.gen_regularization_loss_tracker.update_state(gen_regularization_loss)
      self.gen_mean_tracker.update_state(tf.reduce_mean(tf.stack(gen_mean)))
      self.gen_std_tracker.update_state(tf.reduce_mean(tf.stack(gen_std)))
      self.gen_skew_tracker.update_state(tf.reduce_mean(tf.stack(gen_skew)))
      self.gen_kurt_tracker.update_state(tf.reduce_mean(tf.stack(gen_kurt)))

      reset = tf.reshape(tf.cast(tf.random.uniform((batch_size,)) <= reset_probability, tf.float32), (-1, 1))
      new_noise = tf.random.normal([batch_size, noise_dim])
      self.noise.assign((1-reset)*current_noise + reset*new_noise)

      return {
        "gen_loss": self.gen_loss_tracker.result(), "disc_loss": self.disc_loss_tracker.result(),
        "gen_regularization_loss": self.gen_regularization_loss_tracker.result(),
        "gen_mean": self.gen_mean_tracker.result(),
        "gen_std": self.gen_std_tracker.result(),
        "gen_skew": self.gen_skew_tracker.result(),
        "gen_kurt": self.gen_kurt_tracker.result()}

class CustomCallback(tf.keras.callbacks.Callback):

  def on_epoch_begin(self, epoch, logs):
    if epoch == 0:
      self.model.lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(lr, epochs*batches_per_epoch)

      self.model.gen_optimizer = tf.keras.optimizers.SGD(self.model.lr_decayed_fn)
      self.model.disc_optimizer = tf.keras.optimizers.SGD(self.model.lr_decayed_fn)

      self.model.val_data = self.model.transform_images(next(data_generator()))[0]

      with file_writer.as_default():
        tf.summary.image("In imgs first", self.model.val_data[0,...], step=epoch, max_outputs=num_examples_to_generate)
        tf.summary.image("In imgs second", self.model.val_data[1,...], step=epoch, max_outputs=num_examples_to_generate)
        tf.summary.image("In imgs third", self.model.val_data[2,...], step=epoch, max_outputs=num_examples_to_generate)
        tf.summary.image("In imgs middle", self.model.val_data[int(self.model.val_data.shape[0]/2),...], step=epoch, max_outputs=num_examples_to_generate)
        tf.summary.image("In imgs end", self.model.val_data[-1,...], step=epoch, max_outputs=num_examples_to_generate)
        
    with file_writer.as_default():
      tf.summary.scalar('lr', self.model.lr_decayed_fn(epoch*batches_per_epoch), step=epoch)
    
    self.model.gen_loss_tracker.reset_states()
    self.model.disc_loss_tracker.reset_states()
    self.model.gen_regularization_loss_tracker.reset_states()
    self.model.gen_mean_tracker.reset_states()
    self.model.gen_std_tracker.reset_states()
    self.model.gen_skew_tracker.reset_states()
    self.model.gen_kurt_tracker.reset_states()

  def on_epoch_end(self, epoch, logs=None):
    all_generated_images, all_noises = self.model.inference()

    first = all_generated_images[0]
    first_noise = all_noises[0]     

    second = all_generated_images[1]
    second_noise = all_noises[1]          

    third = all_generated_images[2]
    third_noise = all_noises[2]          

    middle = all_generated_images[int(len(all_generated_images)/2)]
    middle_noise = all_noises[int(len(all_noises)/2)]          

    end = all_generated_images[-1]
    end_noise = all_noises[-1]          

    with file_writer.as_default():
      tf.summary.image("Out imgs first", first, step=epoch, max_outputs=num_examples_to_generate)
      tf.summary.image("Out imgs second", second, step=epoch, max_outputs=num_examples_to_generate)
      tf.summary.image("Out imgs third", third, step=epoch, max_outputs=num_examples_to_generate)
      tf.summary.image("Out imgs middle", middle, step=epoch, max_outputs=num_examples_to_generate)
      tf.summary.image("Out imgs end", end, step=epoch, max_outputs=num_examples_to_generate)

      tf.summary.histogram("Out noise first", tf.reshape(first_noise, (-1,)), step=epoch)
      tf.summary.histogram("Out noise second", tf.reshape(second_noise, (-1,)), step=epoch)
      tf.summary.histogram("Out noise third", tf.reshape(third_noise, (-1,)), step=epoch)
      tf.summary.histogram("Out noise middle", tf.reshape(middle_noise, (-1,)), step=epoch)
      tf.summary.histogram("Out noise end", tf.reshape(end_noise, (-1,)), step=epoch)

      for key in logs:
        tf.summary.scalar(key, logs[key], step=epoch)

model = CustomModel()
model.compile()#, run_eagerly=True)
if args.weights != '':
  
  print("Loading weights from", args.weights)
  model.load_weights(args.weights)

if args.mode == 'train':
  x_files = sorted(glob(f'{args.data_dir}/*.png'))[:n_items]
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

  logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  file_writer = tf.summary.create_file_writer(logdir)
  model.fit(x=data_generator(),
            epochs=epochs,
            callbacks=[
              CustomCallback(), 
              tf.keras.callbacks.ModelCheckpoint(
                os.path.join(logdir, "weights.{epoch:02d}"), verbose=1, save_weights_only=True, save_freq=10*batches_per_epoch)
            ],
            shuffle=False,
            steps_per_epoch=batches_per_epoch)

elif args.mode == 'live' or args.mode == 'video':
  noise = tf.random.normal([batch_size, noise_dim])
  original_fps = int(original_lava_lamp_fps/every_nth)
  num_of_frames = int(output_fps/original_fps)
  print('output_fps', output_fps, 'num_of_frames', num_of_frames)
  frame = None
  last_frame = None
  if args.mode == 'video':
    p = subprocess.Popen(f'ffmpeg -y -f image2pipe -vcodec png -r {output_fps} -i - -f apng -plays 0 -r {output_fps} out.png'.split(' '), stdin=subprocess.PIPE)
    # p = subprocess.Popen(f'ffmpeg -y -f image2pipe -vcodec png -r {fps} -i - -f mp4 -vcodec libx264 -plays 0 -pix_fmt yuv420p -r {fps} -crf 1 out.mp4'.split(' '), stdin=subprocess.PIPE)
  elif args.mode == 'live':
    evaluation_duration = 1000000
  for i in range(evaluation_duration*original_fps):
    start_time = time.time()
    last_frame = frame
    frame, noise = model.generate(noise)
    gen_mean = tf.reduce_mean(noise)
    gen_std = tf.math.reduce_std(noise)
    noise = (noise - gen_mean)/gen_std
    frame = tf.image.convert_image_dtype(frame[0,...], tf.uint8)
    frame = frame.numpy().squeeze()
    if i > 0:
      if start_time is None:
        start_time = time.time()
      frame_float = frame.astype(np.float32)
      last_frame_float = last_frame.astype(np.float32)
      for j in range(num_of_frames):
        interpolated_frame = (last_frame_float * ((num_of_frames-j)/num_of_frames) + frame_float * (j/num_of_frames)).astype(np.uint8)
        print('real frame', i, 'interpol. frame', i*num_of_frames+j, end='\r')
        if args.mode == 'video':
          im = Image.fromarray(interpolated_frame)
          im.save(p.stdin, 'PNG')
        elif args.mode == 'live':
          interpolated_frame = cv2.cvtColor(interpolated_frame, cv2.COLOR_BGR2RGB)
          time.sleep(max((1/output_fps) - (time.time() - start_time), 0))
          cv2.imshow('', interpolated_frame)  
          if cv2.waitKey(1) & 0xFF == ord('q'):  
              break
    start_time = None

  if args.mode == 'video':
    p.stdin.close()
    p.wait()

