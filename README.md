# MANN - Multimodal Attention-based Neural Network
This is a part of Bittersweet's Graduation project.
This repo implements the MANN framework (brought up
by [this paper](https://dl.acm.org/citation.cfm?id=3219819.3219960]))
using TensorFlow.

## PIP dependencies
- tensorflow (or tensorflow-gpu)
- matplotlib

## Project structure
- dev: toys while learning TensorFlow eager execution,
    MANN implemented with tf.enable_eager_execution()
    - training: `cd dev && python train.py`
    - inference: `cd dev && python inference.py`
    - hyper parameters are read from dev/hyper_params.py
    
## See also
- [LeetCode Spider](https://github.com/zhouziqunzzq/GP-leetcode-spider)