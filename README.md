# Artificial Intelligence Resources

## Machine Learning & Deep Learning

### Feature Engineering

The classical/traditional way of "massage" the input to pass to a ML model (e.g. a classifier.) This was
an "art" in itself.
In the age of DL, this has been substituted by the DL models themselves, which represents
also the features on top of which the learning of a task is done. 

## Books

## Frameworks

### Deep Learning
* [Pytorch](https://pytorch.org/)
Recent trend in Research Community. Improvig a lot to put into production
* [Tensorflow/Keras](https://www.tensorflow.org/)
The big hit in 2015.
### NLP
[Huggingface](https://huggingface.co/)

### NLP Papers

#### Recent Origins

* [Word2Vec]()



#### Transformers

#### Sesame Street Environment

* [ELMO](https://arxiv.org/abs/1802.05365) Improvement over Word2Vec. Tries to add context to word representations.
Word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), pre-trained on a large text corpus.
* [BERT](https://arxiv.org/abs/1810.04805) The reference model for NLP since 2018.
* [SpanBERT](https://arxiv.org/pdf/1907.10529.pdf) 
 Masks spans of words instead of random subwords. Spans of words refers to global entities or loca/domain-specific meaning (e.g. American Football)
 Span Boundary Objective(SBO) predicts the span context from boundary token representations. Uses single sentence document-level inputs instead of 
 the two sentences in BERT.
 Code: https://github.com/facebookresearch/SpanBERT
* [RoBERTa](https://arxiv.org/abs/1907.11692)
Replication study of BERT pretraining that measures the impact of many key hyperparameters (Bigger Batch size and LR) and training data size (10X).
It shows improvements on most of the SotA results by BERT and followers. Questions the results of some post-BERT models.
It uses a single sentence for the document-level input like SpanBERT. 
Code: https://github.com/pytorch/fairseq

<details>
  <summary>Small Models</summary>  
 * [Distilbert](https://arxiv.org/abs/1910.01108)
 Check it out in https://huggingface.co/  
</details>

<details>
  <summary>Other Sesame Street Papers</summary>
 
 * [FinBERT](https://arxiv.org/abs/1908.10063) Bert applied to Financial Sentiment Analysis.
 Code: https://github.com/ProsusAI/finBERT]
    
</details>

## Topics

### Transfer Learning
Recent success of DL has been produced, among other reasons, for the big amount of labeled training data.
However, in general, this is not the approach to follow for AI solving different tasks.
Humans are good at learning from very few examples, so scientifically there's still "something" we still need to understand.

Approach of modern Transfer-Learning: 
1. Pre-train in large genral corpus, usually unsupervised (e.g. as in BERT)
1. Fine-Tunning on specific tasks with smaller (supervised) training sets.

Approach of Hierarchical-Multilevel Classification:
* [Hierarchical Transfer Learning for Multi-label Text Classification](https://www.aclweb.org/anthology/P19-1633/)

#### Few-shot Learning (Meta-learning)

Meta-learning aims to train a general model in several learning tasks. The goal is that the resulting model has to be able to solve unseen tasks by using just 
a few training examples.

Concept of [shortcut learning]: You don't learn a task by completely understanding it but by taking "shortcuts" imitating.
e.g. when training on some math exercises in highschool because every year they had the same structure. 

* [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780)

Meta-Learning -learning (how) to learn-: Find an algorithm <img src="https://render.githubusercontent.com/render/math?math=A"> that from a small input data (few-shot examples) <img src="https://render.githubusercontent.com/render/math?math=DS_{train}(x_i,y_i)"> can predict the output <img src="https://render.githubusercontent.com/render/math?math=y'"> of a new input 
<img src="https://render.githubusercontent.com/render/math?math=x'">

* Prototypical Networks [Prototypical Networks for Few-shot Learning]()
Nearest centroid classification

* MAML [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks]() 
Model-agnostic meta-learning algorithm compatible with any model trained with GD and applicable to a variety of different learning problems, including 
classification, regression, and reinforcement learning.

* [Selecting Relevant Features from a Multi-Domain Representation for Few-shot Learning]()

Work from Hugo Larochelle at Google Brain:
* [Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples]()
* [A Universal Representation Transformer Layer for Few-shot Image Classification]()

### Calibration

A measure of the confidence of the predictions of a model. Concept comes from weather forecast.

Main Methods:

- Platt Scaling
- Matrix Vector Scaling
- Temperature scaling

Papers:
* [Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration](https://papers.nips.cc/paper/9397-beyond-temperature-scaling-obtaining-well-calibrated-multi-class-probabilities-with-dirichlet-calibration.pdf)
* [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)

<details>
  <summary>Videos</summary>
 * [Calibration Tutorial] (https://www.youtube.com/watch?v=rhnqZV6eKlg&feature=youtu.be)
</details>


### Causality

* [The Book of Why (Judea Perl)](http://bayes.cs.ucla.edu/WHY/)
* [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)
Learning paradigm to estimate invariant correlations across multiple training distributions. IRM learns a data representation such that the optimal classifier, 
on top of that data representation, matches for all training distributions.

## Main Conferences

* [ACL](https://www.aclweb.org/)
* [KDD (SIGKDD) ACM](https://www.kdd.org/)