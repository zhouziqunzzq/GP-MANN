#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: harry
# @Date  : 2019/4/12 下午2:59
# @Desc  : MANN training with eager execution enabled

import tensorflow as tf
import matplotlib.pyplot as plt
import os
from callbacks import MyModelCheckpoint, PrintLossCallback
from mann import MANNModel
from constants import *
from hyper_params import *
from utils import convert_sparse

tf.enable_eager_execution()


# define parse functions
def _parse_function(example_proto):
    features = {
        'Text': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'Tokens': tf.VarLenFeature(dtype=tf.int64),
        'Tags': tf.VarLenFeature(dtype=tf.int64),
        'Similar Question Text': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'Similar Question Tokens': tf.VarLenFeature(dtype=tf.int64),
        'Similar Question Tags': tf.VarLenFeature(dtype=tf.int64),
        'Dissimilar Question Text': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'Dissimilar Question Tokens': tf.VarLenFeature(dtype=tf.int64),
        'Dissimilar Question Tags': tf.VarLenFeature(dtype=tf.int64),
    }
    parsed_features = tf.parse_single_example(serialized=example_proto, features=features)
    convert_sparse(parsed_features, [
        'Tokens',
        'Tags',
        'Similar Question Tokens',
        'Similar Question Tags',
        'Dissimilar Question Tokens',
        'Dissimilar Question Tags',
    ])
    return (
        parsed_features['Tokens'],
        parsed_features['Similar Question Tokens'],
        parsed_features['Dissimilar Question Tokens']
    )


# custom loss function
def triplet_loss(y_true, y_pred):
    # calculate triplet loss
    anchor_pos, anchor_neg = tf.split(y_pred, 2, axis=-1)
    loss = tf.math.maximum(0.0, MARGIN - anchor_pos + anchor_neg)
    loss = tf.math.reduce_sum(loss, axis=0)
    return loss


def main():
    print("Enable Eager Execution: {}".format(tf.executing_eagerly()))
    # prepare data
    dataset = tf.data.TFRecordDataset(filenames=TRAINING_DATA_FILE_LIST)
    dataset = dataset.map(_parse_function)
    dummy_dataset = tf.data.Dataset.from_tensor_slices(tf.constant([0], dtype=tf.int64))
    dataset = dataset.zip(datasets=(dataset, dummy_dataset))
    dataset = dataset.shuffle(buffer_size=tf.constant(SHUFFLE_BUFFER_SIZE, dtype=tf.int64))
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)

    # take a glance at what the dataset looks like
    for x in dataset.take(1):
        print(type(x))
        print(repr(x))

    # build model
    model = MANNModel(
        vocab_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_SIZE,
        lstm_units=LSTM_UNITS,
        dense_units=DENSE_UNITS,
        training=True,
    )
    # model.build(input_shape=[(1, 200), (1, 200), (1, 200)])
    # model.summary()
    model.compile(
        optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE),
        loss=triplet_loss,
        metrics=['accuracy'],
    )

    # load saved weight if checkpoint file exists
    file_cnt = len([name for name in os.listdir(SAVE_PATH) if os.path.isfile(os.path.join(SAVE_PATH, name))])
    if file_cnt > 0:
        print("Loading model from checkpoint file {}".format(SAVE_FILE))
        model.load_weights(SAVE_FILE)
    else:
        print("No checkpoint file found, training from scratch")

    # instantiate callback
    print_loss_callback = PrintLossCallback()

    # train model
    history = model.fit(
        dataset,
        epochs=100,
        steps_per_epoch=100,
        batch_size=None,
        callbacks=[
            print_loss_callback,
            MyModelCheckpoint(
                filepath=SAVE_FILE,
                best_filepath=SAVE_BEST_FILE,
                monitor='loss',
                verbose=1,
                save_best_only=False,
                save_weights_only=True,
                mode='auto',
                period=1,
            )],
    )

    # save weights
    model.save_weights(SAVE_FILE)

    # plot loss
    print(repr(history.history['loss']))
    plt.plot(history.history['loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()

    return


if __name__ == "__main__":
    main()
