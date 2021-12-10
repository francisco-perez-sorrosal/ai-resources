# Algorithms and Model Architectures

## Classical models

These models work with [features extracted](vocabulary.md#feature-engineering) from the data available. However, because
of this, these models can't take full advantage of large amounts of training data.

### Naive Bayes

### Support Vector Machines

[SVMs](https://nlp.stanford.edu/IR-book/html/htmledition/support-vector-machines-and-machine-learning-on-documents-1.html) 
were co-invented by [Vladimir Vapnik](https://en.wikipedia.org/wiki/Vladimir_Vapnik) and Alexey Ya. 
Chervonenkis in the 60s. While in AT&T during the 90s, Vapnik and other colleagues developed the so-called "kernel 
trick" to create nonlinear classifiers.

SVMs tries to find a decision boundary for data clusters that maximizes the clearance among them (what is called the 
[margin](https://nlp.stanford.edu/IR-book/html/htmledition/support-vector-machines-the-linearly-separable-case-1.html)), 
and it does the best it can when there are outliers.

For datasets that in principle, cannot be separated linearly, the [kernel trick](https://en.wikipedia.org/wiki/Kernel_method) allows to change from linear to 
non-linear decision surfaces.

### Hidden Markov Models

HMMs

### Gradient Boosting Trees

GBTs

### Random Forests

## Neural Network Approaches

These neural network based algorithms to try to overcome the limitations of feature engineering over the data.
For example in NLP, these algorithms try to map input text to a low-dimensional continuous vector, which during
 training, captures the patterns that conform the distinct "features" or patterns hidden in that input text.