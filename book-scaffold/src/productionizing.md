# Productionizing ML/DL

[Multimodal Learning with Incomplete Modalities by Knowledge Distillation (KDD 2020)](http://pages.cs.wisc.edu/~wentaowu/papers/kdd20-ci-for-ml.pdf)
Interesting

## Model Size vs Efficiency
Big models -> Big problem for company at deploy time. Not to speak about deploying an ensemble of models, even if this
shows better performance overall. Several techniques, such as knowledge distillation, pruning and quantization, have 
been identified to reduce the number of parameters of a model without impacting significantly the quality of the model. 
In the end, most of the techniques described below, result in slightly degraded prediction metrics.

* [How We Scaled Bert To Serve 1+ Billion Daily Requests on CPUs (Roblox)](https://blog.roblox.com/2020/05/scaled-bert-serve-1-billion-daily-requests-cpus/)

### Distilation

Hinton, Vinyals and Dean showed in {{#cite hinton_distilling_2015}} how to apply Caruana's model compression techniques described in {{#cite bucilua_model_2006}}.
Caruana et all showed how to take advantage of the property of ANN of being [universal approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem),
to train an ANN to mimic the function learned by an ensemble of models. The idea behind the universal approximator theorem
is that, with enough neurons and training data, a NN can approximate any function with enough precision. To do that,
basically they take a brand new (and usually big) *unlabeled* dataset and they label it using the ensemble. Then they
train an ANN using this brand new large (and recently labeled dataset,) so the resulting model mimics the ensemble, and
which, as they demonstrate, performs much better than the same ANN trained on the original dataset. 
 
Hinton et all, in the aforementioned paper, prove Caruana's ensemble model distillation on MNIST and in a commercial
acoustic model. They also add a new composite ensemble with several specialist models (which can also be trained in 
parallel) that learn to distinguish classes that the full models confuse.

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
* [Training BERT with Deep Speed](https://www.youtube.com/watch?v=n4bESjZ-VaY&feature=youtu.be)
* [Torch Elastic](https://pytorch.org/elastic)
* [PyTorch RPC](https://pytorch.org/docs/stable/rpc.html)
* [PyTorch Serve](https://pytorch.org/serve)
