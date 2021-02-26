"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import struct
import tensorflow as tf
from PIL import Image  
import numpy as np
import PIL
import cv2


FLAGS = tf.app.flags.FLAGS

def PIL_resize(image, ratio, mode):
  PIL_image = PIL.Image.fromarray(image.astype(dtype=np.uint8))
  PIL_image_resize = PIL_image.resize((int(PIL_image.size[0] * ratio), int(PIL_image.size[1] * ratio)), mode)
  image_resize = (np.array(PIL_image_resize)).astype(dtype=np.uint8)
  return image_resize

def imread(path):
    img = cv2.imread(path)
    return img


def read_data(path):
  """
  Read h5 format data file
  
  Args:
    path: file path of desired file

  Returns:
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def modcrop(img, scale =3):
    if len(img.shape) ==3:
        h, w, _ = img.shape
        h = int((h / scale)) * scale
        w = int((w / scale)) * scale
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = int((h / scale)) * scale
        w = int((w / scale)) * scale
        img = img[0:h, 0:w]
    return img


def preprocess(path, scale = 3, eng = None, mdouble = None):
    img = imread(path)
    label_ = modcrop(img, scale)
    if eng is None:
        # input_ = cv2.resize(label_, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_CUBIC)
        input_ = PIL_resize(label_, 1.0/scale, PIL.Image.BICUBIC)
    else:
        input_ = np.asarray(eng.imresize(mdouble(label_.tolist()), 1.0/scale, 'bicubic'))

    input_ = input_[:, :, ::-1]
    label_ = label_[:, :, ::-1]

    return input_, label_



def prepare_data(sess, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  if FLAGS.train:
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
  else:
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
  data = sorted(glob.glob(os.path.join(data_dir, "*.bmp")))

  return data

def make_data(sess, checkpoint_dir, data, label):
  """
  Make input data as h5 file format
  Depending on 'train' (flag value), savepath would be changed.
  """
  if FLAGS.train:
    savepath = os.path.join(os.getcwd(), '{}/train.h5'.format(checkpoint_dir))
  else:
    savepath = os.path.join(os.getcwd(), '{}/test.h5'.format(checkpoint_dir))

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)

def image_read(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return imread(path, mode='YCbCr').astype(np.float)


def train_input_worker(args):
  image_data, config = args
  image_size, label_size, stride, scale, save_image = config

  single_input_sequence, single_label_sequence = [], []
  padding = abs(image_size - label_size) / 2 # eg. for 3x: (21 - 11) / 2 = 5
  label_padding = label_size / scale # eg. for 3x: 21 / 3 = 7

  input_, label_ = preprocess(image_data, scale)

  if len(input_.shape) == 3:
    h, w, _ = input_.shape
  else:
    h, w = input_.shape

  for x in range(0, h - image_size - padding + 1, stride):
    for y in range(0, w - image_size - padding + 1, stride):
      sub_input = input_[x + padding : x + padding + image_size, y + padding : y + padding + image_size]
      x_loc, y_loc = x + label_padding, y + label_padding
      sub_label = label_[x_loc * scale : x_loc * scale + label_size, y_loc * scale : y_loc * scale + label_size]

      sub_input = sub_input.reshape([image_size, image_size, 1])
      sub_label = sub_label.reshape([label_size, label_size, 1])
      
      single_input_sequence.append(sub_input)
      single_label_sequence.append(sub_label)

  return [single_input_sequence, single_label_sequence]

def train_input_setup(config):
  """
  Read image files, make their sub-images, and save them as a h5 file format.
  """
  sess = config.sess
  image_size, label_size, stride, scale = config.image_size, config.label_size, config.stride, config.scale

  # Load data path
  data = prepare_data(sess, dataset=config.data_dir)

  sub_input_sequence, sub_label_sequence = [], []
  padding = abs(image_size - label_size) / 2 # eg. for 3x: (21 - 11) / 2 = 5
  label_padding = label_size / scale # eg. for 3x: 21 / 3 = 7

  for i in range(len(data)):
    input_, label_ = preprocess(data[i], scale)

    if len(input_.shape) == 3:
      h, w, _ = input_.shape
    else:
      h, w = input_.shape

    for x in range(0, h - image_size - padding + 1, stride):
      for y in range(0, w - image_size - padding + 1, stride):
        sub_input = input_[x + padding : x + padding + image_size, y + padding : y + padding + image_size]
        x_loc, y_loc = x + label_padding, y + label_padding
        sub_label = label_[x_loc * scale : x_loc * scale + label_size, y_loc * scale : y_loc * scale + label_size]

        sub_input = sub_input.reshape([image_size, image_size, 1])
        sub_label = sub_label.reshape([label_size, label_size, 1])
        
        sub_input_sequence.append(sub_input)
        sub_label_sequence.append(sub_label)

  arrdata = np.asarray(sub_input_sequence)
  arrlabel = np.asarray(sub_label_sequence)

  make_data(sess, config.checkpoint_dir, arrdata, arrlabel)


def test_input_setup(config):
  """
  Read image files, make their sub-images, and save them as a h5 file format.
  """
  sess = config.sess
  image_size, label_size, stride, scale = config.image_size, config.label_size, config.stride, config.scale

  # Load data path
  data = prepare_data(sess, dataset="Test")

  sub_input_sequence, sub_label_sequence = [], []
  padding = abs(image_size - label_size) / 2 # eg. (21 - 11) / 2 = 5
  label_padding = label_size / scale # eg. 21 / 3 = 7

  pic_index = 2 # Index of image based on lexicographic order in data folder
  input_, label_ = preprocess(data[pic_index], config.scale)

  if len(input_.shape) == 3:
    h, w, _ = input_.shape
  else:
    h, w = input_.shape

  nx, ny = 0, 0
  for x in range(0, h - image_size - padding + 1, stride):
    nx += 1
    ny = 0
    for y in range(0, w - image_size - padding + 1, stride):
      ny += 1
      sub_input = input_[x + padding : x + padding + image_size, y + padding : y + padding + image_size]
      x_loc, y_loc = x + label_padding, y + label_padding
      sub_label = label_[x_loc * scale : x_loc * scale + label_size, y_loc * scale : y_loc * scale + label_size]

      sub_input = sub_input.reshape([image_size, image_size, 1])
      sub_label = sub_label.reshape([label_size, label_size, 1])
      
      sub_input_sequence.append(sub_input)
      sub_label_sequence.append(sub_label)

  arrdata = np.asarray(sub_input_sequence)
  arrlabel = np.asarray(sub_label_sequence)

  make_data(sess, config.checkpoint_dir, arrdata, arrlabel)

  return nx, ny

# You can ignore, I just wanted to see how much space all the parameters would take up
def save_params(sess, weights, biases):
  param_dir = "params/"

  if not os.path.exists(param_dir):
    os.makedirs(param_dir)

  weight_file = open(param_dir + "weights", 'wb')
  for layer in weights:
    layer_weights = sess.run(weights[layer])

    for filter_x in range(len(layer_weights)):
      for filter_y in range(len(layer_weights[filter_x])):
        filter_weights = layer_weights[filter_x][filter_y]
        for input_channel in range(len(filter_weights)):
          for output_channel in range(len(filter_weights[input_channel])):
            weight_value = filter_weights[input_channel][output_channel]
            # Write bytes directly to save space 
            weight_file.write(struct.pack("f", weight_value))
          weight_file.write(struct.pack("x"))

    weight_file.write("\n\n")
  weight_file.close()

  bias_file = open(param_dir + "biases.txt", 'w')
  for layer in biases:
    bias_file.write("Layer {}\n".format(layer))
    layer_biases = sess.run(biases[layer])
    for bias in layer_biases:
      # Can write as characters due to low bias parameter count
      bias_file.write("{}, ".format(bias))
    bias_file.write("\n\n")

  bias_file.close()

def merge(images, size):
  """
  Merges sub-images back into original image size
  """
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 1))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img

def array_image_save(array, image_path):
  """
  Converts np array to image and saves it
  """
  image = Image.fromarray(array)
  if image.mode != 'RGB':
    image = image.convert('RGB')
  image.save(image_path)
  print("Saved image: {}".format(image_path))
