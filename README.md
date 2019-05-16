# MANN - Multimodal Attention-based Neural Network
This is a part of Bittersweet's Graduation project.
This repo implements the MANN framework (brought up
by [this paper](https://dl.acm.org/citation.cfm?id=3219819.3219960]))
using TensorFlow.

## PIP dependencies
- tensorflow (or tensorflow-gpu)
- sklearn
- numpy
- matplotlib
- lxml

## Project structure
- MANN_naive: MANN implemented without any attention
    - using dummy data to test convergence: `python train_test.py`
    - training: `python train.py`
    - inference: `python inference.py`
    - evaluate: `python evaluate.py`
    - hyper parameters are read from hyper_params.py
    - constants are read from constants.py
    
- MANN_SA: MANN implemented with SA(Similarity Attention)
    - using dummy data to test convergence: `python train_test.py`
    - training: `python train.py`
    - inference: `python inference.py`
    - evaluate: `python evaluate.py`
    - hyper parameters are read from hyper_params.py
    - constants are read from constants.py
    
- MANN_TCA: MANN implemented with TCA(Text-Concept Attention)
    - training: `python train.py`
    - inference: `python inference.py`
    - evaluate: `python evaluate.py`
    - hyper parameters are read from hyper_params.py
    - constants are read from constants.py
    
- MANN_TCA_cudnn: MANN implemented with TCA(Text-Concept Attention)
written with CuDNNLSTM
    - training: `python train.py`
    - hyper parameters are read from hyper_params.py
    - constants are read from constants.py
    
- MANN_TCA_SA: MANN implemented with TCA(Text-Concept Attention)
and SA(Similarity Attention)
    - training: `python train.py`
    - inference: `python inference.py`
    - evaluate: `python evaluate.py`
    - hyper parameters are read from hyper_params.py
    - constants are read from constants.py
    
- LCS: Least Common Sub-sequence
    - inference: `python inference.py`
    - evaluate: `python evaluate.py`
    
- TF-IDF: Term Frequency - Inverse Document Frequency
    - inference: `python inference.py`
    - evaluate: `python evaluate.py`
    
## See also
- [LeetCode Spider](https://github.com/zhouziqunzzq/GP-leetcode-spider)