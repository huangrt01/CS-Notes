'''
1 * V100 Images/sec: 1312
4 * V100 Images/sec: 4960
并未得到完美的线性增加，其中的一个重要原因是由于更新权重时GPU之间的通信，也可能有其它因素，例如在权重平均之前等待较慢的GPU完成处理
'''

from __future__ import print_function

import argparse
import keras
from keras import backend as K
from keras.preprocessing import image
from keras.datasets import fashion_mnist
from keras_contrib.applications.wide_resnet import WideResidualNetwork
import numpy as np
import tensorflow as tf
import os
from time import time
import csv
import horovod.keras as hvd
from novograd import NovoGrad

# Initialize Horovod
hvd.init()

# Pin to a GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

# Parse input arguments

parser = argparse.ArgumentParser(description='Keras Fashion MNIST Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--novo-grad', action='store_true',
                    help='whether use NovoGrad as optimizer')
parser.add_argument('--base-lr', type=float, default=0.01,
                    help='learning rate for a single GPU')
parser.add_argument('--wd', type=float, default=0.000005,
                    help='weight decay')
parser.add_argument('--target-accuracy', type=float, default=.91,
                    help='Target accuracy to stop training')
parser.add_argument('--patience', type=float, default=2,
                    help='Number of epochs that meet target before stopping')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')

args = parser.parse_args()

# Define a function for a simple learning rate decay over time

def lr_schedule(epoch):
    
    if epoch < 15:
        return args.base_lr
    if epoch < 25:
        return 1e-1 * args.base_lr
    if epoch < 35:
        return 1e-2 * args.base_lr
    return 1e-3 * args.base_lr

def create_model():

    # Set up standard WideResNet-16-10 model.
    model = WideResidualNetwork(depth=16, width=10, weights=None, input_shape=input_shape,
                                classes=num_classes, dropout_rate=0.01)

    # WideResNet model that is included with Keras is optimized for inference.
    # Add L2 weight decay & adjust BN settings.
    model_config = model.get_config()
    for layer, layer_config in zip(model.layers, model_config['layers']):
        if hasattr(layer, 'kernel_regularizer'):
            regularizer = keras.regularizers.l2(args.wd)
            layer_config['config']['kernel_regularizer'] = \
                {'class_name': regularizer.__class__.__name__,
                 'config': regularizer.get_config()}
        if type(layer) == keras.layers.BatchNormalization:
            layer_config['config']['momentum'] = 0.9
            layer_config['config']['epsilon'] = 1e-5

    model = keras.models.Model.from_config(model_config)

    if args.novo_grad:
        opt = NovoGrad(lr=args.base_lr)
    else:
        opt = keras.optimizers.SGD(lr=args.base_lr, momentum=args.momentum)
    
    # Wrap the optimizer in a Horovod distributed optimizer
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
        
    return model


# Only set `verbose` to `1` if this is the root worker.
# Otherwise, it should be zero.
verbose = hvd.rank() == 0

# Input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

# Load Fashion MNIST data.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Training data iterator.
train_gen = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                     horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2)
train_gen.fit(x_train)
train_iter = train_gen.flow(x_train, y_train, batch_size=args.batch_size)

# Validation data iterator.
test_gen = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
test_gen.mean = train_gen.mean
test_gen.std = train_gen.std
test_iter = test_gen.flow(x_test, y_test, batch_size=args.batch_size)


callbacks = []
callbacks.append(keras.callbacks.LearningRateScheduler(lr_schedule))

# Custom Callbacks
class PrintThroughput(keras.callbacks.Callback):
    def __init__(self, total_images=0):
        self.total_images = total_images
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time()
    
    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time() - self.epoch_start_time
        images_per_sec = round(self.total_images / epoch_time, 2)
        print('Images/sec: {}'.format(images_per_sec))

class StopAtAccuracy(keras.callbacks.Callback):
    def __init__(self, target=0.85, patience=2, verbose=0):
        self.target = target
        self.patience = patience
        self.verbose = verbose
        self.stopped_epoch = 0
        self.met_target = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_acc') > self.target:
            self.met_target += 1
        else:
            self.met_target = 0

        if self.met_target >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and verbose == 1:
            print('Early stopping after epoch {}. Target met: {}'.format(self.stopped_epoch + 1, self.target))

class PrintTotalTime(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time()
    
    def on_epoch_end(self, epoch, logs=None):
        total_time = round(time() - self.start_time, 2)
        print("Total training time: {}".format(total_time)) 
        
# New Callback: SaveTraningData
class SaveTrainingData(tf.keras.callbacks.Callback):
    def __init__(self, data_filepath=''):
        self.data_filepath = data_filepath
        
    def on_train_begin(self, logs=None):       
        file = open(self.data_filepath, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(['time', 'val_accuracy'])
        writer.writerow([0.0, 0.0])
        file.close()  
        
        self.train_start_time = time()
    
    def on_epoch_end(self, epoch, logs={}):
        total_time = time() - self.train_start_time
        file = open(self.data_filepath, 'a')
        writer = csv.writer(file)
        writer.writerow([round(total_time,1), round(logs['val_acc'], 4)])
        file.close()

if verbose:
    callbacks.append(PrintTotalTime()) 
    callbacks.append(PrintThroughput(total_images=len(y_train))) 

callbacks.append(StopAtAccuracy(target=args.target_accuracy, patience=args.patience, verbose=verbose))

# Broadcast initial variable states from the first worker to 
# all others by adding the broadcast global variables callback.
callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

# Average the metrics among workers at the end of every epoch
# by adding the metric average callback.
callbacks.append(hvd.callbacks.MetricAverageCallback())

# Add learning rate warmup callback:
callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose))

# Construct CSV filename from hyperparameters (with momentum)
ranks = hvd.size()
batch_size = args.batch_size
lr = args.base_lr
m = args.momentum
warmups = args.warmup_epochs
o = "-novo" if args.novo_grad else ""
data_filepath = "training_data/{}ranks-{}bs-{}lr-{}m-{}wu{}.csv".format(ranks,batch_size,lr,m,warmups,o)

if hvd.rank() == 0:
    # add SaveTrainingData callback
    callbacks.append(SaveTrainingData(data_filepath=data_filepath))

# Create the model.
model = create_model()
    
# Train the model.
model.fit_generator(train_iter,
                    # Keep the total number of steps the same despite of an increased number of workers
                    steps_per_epoch=len(train_iter) // hvd.size(),
                    callbacks=callbacks,
                    epochs=args.epochs,
                    verbose=verbose,
                    workers=4,
                    initial_epoch=0,
                    validation_data=test_iter,
                    # We set this to 3X so that we are more likely to use every instance in the validation set
                    validation_steps=3 * len(test_iter) // hvd.size())

# Evaluate the model on the full data set.
score = model.evaluate_generator(test_iter, len(test_iter), workers=4)
if verbose:
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])