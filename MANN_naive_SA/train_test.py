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
def merge_function(*args):
    return args


def get_parse_function(feature_name: str):
    assert feature_name in ['Text', 'Tokens', 'Tags', 'Similar Question Text',
                            'Similar Question Tokens', 'Similar Question Tags',
                            'Dissimilar Question Text', 'Dissimilar Question Tokens',
                            'Dissimilar Question Tags']

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
        return parsed_features[feature_name]

    return _parse_function


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
    dataset_tokens = tf.data.Dataset.from_tensor_slices(([[3615, 3615, 3615, 3615, 3615],
                                                          [323, 323, 323, 323, 323]])).repeat()
    dataset_sim_tokens = tf.data.Dataset.from_tensor_slices(([[3615, 3615, 3615, 3615, 3615],
                                                              [323, 323, 323, 323, 323]])).repeat()
    dataset_dis_tokens = tf.data.Dataset.from_tensor_slices(([[323, 323, 323, 323, 323],
                                                              [3615, 3615, 3615, 3615, 3615]])).repeat()

    dataset_tokens = dataset_tokens.padded_batch(BATCH_SIZE, padded_shapes=[SEQ_LENGTH])
    dataset_sim_tokens = dataset_sim_tokens.padded_batch(BATCH_SIZE, padded_shapes=[SEQ_LENGTH])
    dataset_dis_tokens = dataset_dis_tokens.padded_batch(BATCH_SIZE, padded_shapes=[SEQ_LENGTH])

    dataset_zipped = tf.data.Dataset.zip((dataset_tokens, dataset_sim_tokens, dataset_dis_tokens))
    dataset_merged = dataset_zipped.map(merge_function, num_parallel_calls=CPU_CORES)

    dummy_dataset = tf.data.Dataset.from_tensor_slices(tf.constant([0], dtype=tf.int64)) \
        .repeat()

    dataset = tf.data.Dataset.zip(datasets=(dataset_merged, dummy_dataset)) \
        .shuffle(buffer_size=tf.constant(SHUFFLE_BUFFER_SIZE, dtype=tf.int64)) \
        .repeat() \
        .prefetch(PREFETCH_SIZE)

    # take a glance at what the dataset looks like
    # for x in dataset.take(5):
    #     print(type(x))
    #     print(repr(x))
    # return

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
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM),
        loss=triplet_loss,
        metrics=[],
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
