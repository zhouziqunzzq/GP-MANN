#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : callbacks.py
# @Author: harry
# @Date  : 2019/4/22 下午7:02
# @Desc  : Description goes here

import tensorflow as tf
import numpy as np
import logging


# callback to print custom loss at every epoch end
class PrintLossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(" loss:{}".format(logs.get('loss')))


# modified version of ModelCheckpoint Callback
class MyModelCheckpoint(tf.keras.callbacks.Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    Arguments:
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self,
                 filepath,
                 best_filepath='',
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 period=1):
        super(MyModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        if best_filepath == '':
            self.best_filepath = filepath
        else:
            self.best_filepath = best_filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            logging.warning('ModelCheckpoint mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            best_filepath = self.best_filepath.format(epoch=epoch + 1, **logs)

            # save to best_filepath
            current = logs.get(self.monitor)
            if current is None:
                logging.warning('Can save best model only with %s available, '
                                'skipping.', self.monitor)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                       current, best_filepath))
                    self.best = current
                    if self.save_weights_only:
                        self.model.save_weights(best_filepath, overwrite=True)
                    else:
                        self.model.save(best_filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve from %0.5f' %
                              (epoch + 1, self.monitor, self.best))

            if not self.save_best_only:
                # save to filepath
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
