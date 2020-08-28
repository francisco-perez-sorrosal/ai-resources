# Artificial Intelligence Resources

“To predict the behavior of ordinary people in advance, you only have to assume that they will always try to escape a 
disagreeable situation with the smallest possible expenditure of intelligence.”
― Friedrich Nietzsche

## Machine Learning & Deep Learning

### Feature Engineering

The classical/traditional way of "massage" the input to pass to a ML model (e.g. a classifier.) This was
an "art" in itself.
In the age of DL, this has been substituted by the DL models themselves, which represents
also the features on top of which the learning of a task is done. 



## Topics

### Deep Learning Core

* [Retrospective Loss: Looking Back to Improve Training of Deep Neural Networks (KDD 2020)](https://arxiv.org/abs/2006.13593)	
Learns from prior training experiences in the form of DNN model states during to guide weight updates and improve DNN 
training performance. The retrospective loss seeks to ensure that the predictions at a particular training step are more 
similar to the ground truth than to the predictions from a previous training step (which has relatively
poorer performance). As training proceeds, minimizing this loss along with the task-specific loss, encourages the 
network parameters to move towards the optimal parameter state by pushing the training into tighter spaces around the 
optimum. Claims implementation is 3 lines of Pytorch code. Interesting paper

### Transfer Learning
Recent success of DL has been produced, among other reasons, for the big amount of labeled training data.
However, in general, this is not the approach to follow for AI solving different tasks.
Humans are good at learning from very few examples, so scientifically there's still "something" we still need to understand.

Approach of modern Transfer-Learning: 
1. Pre-train in large genral corpus, usually unsupervised (e.g. as in BERT)
1. Fine-Tunning on specific tasks with smaller (supervised) training sets.

Approach of Hierarchical-Multilevel Classification:
* [Hierarchical Transfer Learning for Multi-label Text Classification (ACL 2019)](https://www.aclweb.org/anthology/P19-1633/)
* [Multi-Source Deep Domain Adaptation with Weak Supervision for Time-Series Sensor Data (KDD 2020)](https://arxiv.org/abs/2005.10996)
Interesting

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

### Model Interpretability
Intellegibility and explanation are critical in many domains (health, crime prediction...) [Blog entry](https://medium.com/analytics-vidhya/model-interpretation-with-microsofts-interpret-ml-85aa0ad697ae)


Models as black box methods: Shap, LIME
Glass box modesl: Explainable Boosting Machine (EBMs) are the SotA

Papers:
* [Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/abs/2004.13912)
* [A Unified Approach to Interpreting Model Predictions (NIPS 2017)](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)

Books:
* [Interpreable Machine Learning Book A Guide for Making Black Box Models Explainable (Christoph Molnar, 2020)](https://christophm.github.io/interpretable-ml-book/)

Tools:
* [Lime](https://github.com/marcotcr/lime)
* [Shap](https://github.com/slundberg/shap)
* [Interpret ML](https://github.com/interpretml/interpret) Reference impl of EBMs
* [Shap Tutorial (Includes BERT examples)](https://nbviewer.jupyter.org/github/slundberg/shap/blob/master/notebooks/general/Explainable%20AI%20with%20Shapley%20Values.ipynb)

## Interpretability

* [Captum](https://captum.ai/) For Pytorch

### Causality

* [The Book of Why (Judea Perl)](http://bayes.cs.ucla.edu/WHY/)
* [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)
Learning paradigm to estimate invariant correlations across multiple training distributions. IRM learns a data representation such that the optimal classifier, 
on top of that data representation, matches for all training distributions.


# Production

[Multimodal Learning with Incomplete Modalities by Knowledge Distillation (KDD 2020)](http://pages.cs.wisc.edu/~wentaowu/papers/kdd20-ci-for-ml.pdf)
Interesting

## Model Size vs Efficiency
Big models -> Big problem for company at deploy time. Identify techniques to reduce the number of parameters
without impacting the quality of the model.

* [How We Scaled Bert To Serve 1+ Billion Daily Requests on CPUs (Roblox)](https://blog.roblox.com/2020/05/scaled-bert-serve-1-billion-daily-requests-cpus/)

### Distilation

### Pruning

### Quantization

* Focused on inference
* Focused on small devices/IoT

Papers:

* [Q8BERT: Quantized 8Bit BERT (2019](https://arxiv.org/abs/1910.06188)
* [DYNAMIC QUANTIZATION ON BERT (BETA)](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)

It turns out that quantization is now now possible in ONNX models:

```python
import onnx
from quantize import quantize, QuantizationMode
...
# Load onnx model
onnx_model = onnx.load('XXX_path')
# Quantize following a specific mode from https://github.com/microsoft/onnxruntime/tree/e26e11b9f7f7b1d153d9ce2ac160cffb241e4ded/onnxruntime/python/tools/quantization#examples-of-various-quantization-modes
q_onnx_model = quantize(onnx_model, quantization_mode=XXXXX)
# Save the quantized model
onnx.save(q_onnx_model, 'XXXX_path')
```

[Tensor RT supports the quantized models so, it should work](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work-with-qat-networks)

* [Deep Speed (Microsoft)](https://github.com/microsoft/DeepSpeed) ZeRO redundancy memory optimizer: Addresses the problems with high memory consumption of 
large models with pure data parallelism and the problem of using model parallelism.

# Industry

* [Training BERT with Deep Speed](https://www.youtube.com/watch?v=n4bESjZ-VaY&feature=youtu.be)
* [Torch Elastic](https://pytorch.org/elastic)
* [PyTorch RPC](https://pytorch.org/docs/stable/rpc.html)
* [PyTorch Serve](https://pytorch.org/serve)
