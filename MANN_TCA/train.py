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
from data_utils import load_training_dataset

tf.enable_eager_execution()


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
    dataset = load_training_dataset(TRAINING_DATA_FILE_LIST)

    # take a glance at what the dataset looks like
    # for x in dataset.take(1):
    #     print(type(x))
    #     print(repr(x))
    # return

    # build model
    model = MANNModel(
        vocab_size=VOCAB_SIZE,
        embedding_size_vocab=EMBEDDING_SIZE_VOCAB,
        tag_size=TAG_SIZE,
        embedding_size_tag=EMBEDDING_SIZE_TAG,
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
