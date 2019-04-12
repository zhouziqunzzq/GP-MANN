#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : mann_eager.py
# @Author: harry
# @Date  : 2019/4/12 下午2:59
# @Desc  : MANN with eager execution enabled

import tensorflow as tf

tf.enable_eager_execution()

# constant
VOCAB_SIZE = 4133
BATCH_SIZE = 64

# hyper params
EMBEDDING_SIZE = 100
LSTM_UNITS = 128
DENSE_UNITS = 256
LEARNING_RATE = 0.01
MARGIN = 0.5
REG_LAMBDA = 0.01


# define parse functions
def _convert_sparse(features: dict, entry_list: list):
    for k in entry_list:
        features[k] = tf.sparse.to_dense(features[k])


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
    _convert_sparse(parsed_features, [
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


# define MANN model
class MANNModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_units, dense_units, training=True):
        super(MANNModel, self).__init__(name='MANN_model')
        self.training = training
        self.embedding = tf.keras.layers.Embedding(
            name="embedding",
            input_dim=vocab_size,
            output_dim=embedding_size,
        )
        self.lstm = tf.keras.layers.CuDNNLSTM(
            name="lstm",
            units=lstm_units,
            return_sequences=False,
            return_state=True,
            # unroll=False,
            kernel_regularizer=tf.keras.regularizers.l2(REG_LAMBDA),
        )
        self.dense1 = tf.keras.layers.Dense(
            name="dense_1",
            units=dense_units,
            kernel_regularizer=tf.keras.regularizers.l2(REG_LAMBDA),
        )
        self.dense2 = tf.keras.layers.Dense(
            name="dense_2",
            units=1,
            kernel_regularizer=tf.keras.regularizers.l2(REG_LAMBDA),
        )

    # TODO: how to use this "training" arg ???
    def call(self, inputs, training=None, mask=None):
        def _get_state(x):
            result = self.embedding(x)
            result = self.lstm(result)
            return result[1]

        def _compute_pair(a, b):
            a_state = _get_state(a)
            b_state = _get_state(b)
            result = tf.keras.layers.concatenate(
                inputs=[a_state, b_state],
                axis=-1,
            )
            result = self.dense1(result)
            result = self.dense2(result)
            return result

        training = self.training
        # print("is training: {}".format(training))

        if training:
            print("Building MANN for training...")
            # build training model with triplet loss
            # unpack inputs (expecting 3 tensors in training mode)
            anchor_tokens, pos_tokens, neg_tokens = inputs
            # compute MANN(anchor, pos)
            anchor_pos = _compute_pair(anchor_tokens, pos_tokens)
            # compute MANN(anchor, neg)
            anchor_neg = _compute_pair(anchor_tokens, neg_tokens)
            # return concatenated results for loss computation
            return tf.keras.layers.concatenate(
                inputs=[anchor_pos, anchor_neg],
                axis=-1,
            )
        else:
            print("Building MANN for inference...")
            # build inference model
            # unpack inputs (expecting 2 tensors in trainning mode)
            a_tokens, b_tokens = inputs
            return _compute_pair(a_tokens, b_tokens)


def triplet_loss(y_true, y_pred):
    # calculate triplet loss
    anchor_pos, anchor_neg = tf.split(y_pred, 2, axis=-1)
    loss = tf.math.maximum(0.0, MARGIN - anchor_pos + anchor_neg)
    loss = tf.math.reduce_sum(loss, axis=0)
    return loss


def main():
    print("Enable Eager Execution: {}".format(tf.executing_eagerly()))
    # prepare data
    filenames = ["./tf_data/leetcode_pairwise.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(_parse_function)
    dummy_dataset = tf.data.Dataset.from_tensor_slices(tf.constant([0], dtype=tf.int64))
    dataset = dataset.zip(datasets=(dataset, dummy_dataset))
    dataset = dataset.shuffle(buffer_size=tf.constant(2048, dtype=tf.int64))
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)

    # take a glance at what the dataset looks like
    # for x in dataset.take(1):
    #     print(type(x))
    #     print(repr(x))

    # build model
    model = MANNModel(
        vocab_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_SIZE,
        lstm_units=LSTM_UNITS,
        dense_units=DENSE_UNITS,
    )
    # model.build(input_shape=[(1, 200), (1, 200), (1, 200)])
    # model.summary()
    model.compile(
        optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE),
        loss=triplet_loss,
        metrics=['accuracy'],
    )

    # train model
    history = model.fit(
        dataset,
        epochs=100,
        steps_per_epoch=10,
        batch_size=None,
    )

    # loss acc
    print(repr(history.history['loss']))
    print(repr(history.history['acc']))

    return


if __name__ == "__main__":
    main()
