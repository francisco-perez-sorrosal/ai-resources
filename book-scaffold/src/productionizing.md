# Productionizing ML/DL

The software infrastructure stack that is required for ML projects in general, is quite different from the current stack
of non-ML projects. [Andrej Karpathy](people.md#Andrej_Karpathy)

[Multimodal Learning with Incomplete Modalities by Knowledge Distillation (KDD 2020)](http://pages.cs.wisc.edu/~wentaowu/papers/kdd20-ci-for-ml.pdf)
Interesting

## Challenges in Putting a ML Model into Production

- __Infrastructure and Tooling__: As of 2021, current tools and platforms, mainly CI/CD do not have great support for data and/or
ML specific computing resources (e.g. GPUs/TPUs) which causes problems when trying to integrate the tasks and necessary
steps that conform a modern ML pipeline for any use case at hand.
- __Culture and Organizational Structure__: Companies should reorganize their divisions to accommodate this new way of producing
software systems based on a ML model. The standard model for software development does not fit here. But that doesn't mean
that data scientists should not adopt norms, techniques and tools from the traditional software development (such as 
testing) and adapt them to their needs.
- __Decision Making__: How to decide that a model is "good enough"? Check [Interpretability](topics.md#Model_Interpretability) 

# Data

## Data Management At Scale

Data management is one of the new problems many companies in the industry are facing. Usually managed by individual teams
in isolation and with very different policies in terms of versioning, privacy and accesibility and location, with the
increasing sources of data, this has become a problem in modern companies.

In the past, solutions like Data Warehouses -built out of multiple of ETL processes- or its evolution, the Data Lake where the norm. 
The trend in 2021 is to evolve the Data Lake view towards the so-called [Data Mesh](https://martinfowler.com/articles/data-monolith-to-mesh.html) are the tendency.
The idea is to scale up the 

## Datasets

It's well know that the main datasets involved in any ML process for creating a model are:

- __Train__, which is the data your model will train on
- __Dev__, which is a representation of the test dataset used to do hyperparameter tuning, select features or make any other
decision over the model. In scikit-learn use to be called __hold-out cross-validation set__. 
- __Test__, which is used exclusively for evaluating the metrics selected for the model. It should not be used to make any other
decision about the process of developing the final model.

However there's more things to take into account, mainly: 

- Main test set should mirror your online distribution
- It's possible to collect alternative datasets
  - They don't have to come from the real distribution but will be useful to evaluate the model selected
  - When is this useful?
    - when there are certain edge cases that we need to actuate on
    - when the model has to run on multiple datasets with different modalities (e.g. English and Spanish datasets, wild and pet animals datasets)
- Compare new model against:
 - 1) the previous model and 
 - 2) a fixed older model (baseline)
 - 3) against the possible slices interesting for the business model
 - 4) against the alternative datasets

More information about dataset creation and quality can be found in the online book by Andrew Ng, [Machine Learning Yearning](https://d2wvfoqc9gyqzf.cloudfront.net/content/uploads/2018/09/Ng-MLY01-13.pdf)

## Versioning

One of the main problems that I found when I started working in the ML field 3 years ago was that there was almost no 
control of what data was used in some of experiments we were doing; the vast majority 
(if not all) of my colleagues were not using versioned data; they were just relying on the knowledge transmitted by the
old members of the team. As an outsider at that time coming from other computer science
disciplines and being more aware of software engineering techniques, that shocked me because I knew that working in a
context like that was to bring two well-known major pains for me:

- Back to old same problems back in the days for not having regular versioned code
- Frustration when trying to communicate and explain that data versioning was helpful and we had to implement a policy
and use tools for keep track of the evolution of the data in our datasets 

But there's more when you introduce data versioning in your scope. Almost immediately it comes up to your mind that you
also have to keep track of the data-code version pairs to be able to reproduce exactly your experiments.

The next step is of course, find tools about version control for data. You can think immediately on Git/Github and similar
tools (Bitbucket, Gitlab, etc..) Sooner that later that those existing solutions might work for projects with
small datasets, but scalability is gonna become a problem (e.g. you can't store in git files bigger than a certain size).
Also because when building a dataset, you need certain flexibility
to manage the deltas you are adding; if you add data incrementally at some point you may create biases (e.g. by 
adding at some point many examples of a particular category in data for a classifier). It would be nice to use a tool that 
despite recognizing you added data to a dataset incrementally, it would allow you certain flexibility when building the datasets
for training/testing based on the deltas added (e.g. by skipping certain delta increments, combining only the base datasets
and two specific deltas, etc). This kind of tools that allow user to create versioned datasets and then aggregate them
by following a workflow-based data combination approach is what it would be nice to have to manage data used in ML model/experiments.

In Lecture 8, Slide 76 of the [Full Stack DL of 2021] there a summary of the approaches that currently are followed by the practitioners:

- Level 0 - No data versioning. Unfortunately, the most common approach.   
- Level 1 - Snapshot at training time. The next level if you are a little bit more careful when developing a new model.
- Level 2 - Data is versioned as mix of assets and code. I pushed hard to follow this approach at least when I implemented my first ML training pipeline a couple of years ago.
- Level 3 - New tools for integrated data versioning. An active space for developing new tools.

### Tools

- 
- [dvc Data Version Control](https://dvc.org/)



# Machine Learning Platforms and Pipelines

At some point, the main leaders in the software development industry decided to show off their ML platforms. Here there
are some of them.

## Architectural Examples

- [Michelangelo (Uber)](https://eng.uber.com/michelangelo-machine-learning-platform/) - Uber's machine learning platform. Here, 
[there's another blogpost](https://eng.uber.com/scaling-michelangelo/) with more details.
- [Metaflow (Netflix)](https://www.youtube.com/watch?v=XV5VGddmP24) - They don't offer much details about the data-related
tasks (e.g. data-lake, versioning, preprocessing, etc.) available in the platform. Post about [moving Metaflow to an open-source effort](https://netflixtechblog.com/open-sourcing-metaflow-a-human-centric-framework-for-data-science-fa72e04a5d9)
- [Pro-ML (Linkedin)](https://engineering.linkedin.com/blog/2019/01/scaling-machine-learning-productivity-at-linkedin) - Also
they don't provide much detail about the data part, which present as a silo on the left part of their architectural diagram,
and called Feature Marketplace.  

## Regular SW vs Machine Learning SW

| Traditional SW | Machine Learning |
| -------------- | ---------------- |
| Code + Config Data | Code + Data + Workflow + Config Data    |
| Written by humans to be validated essentially by humans | Specified by humans, optimized by compilers to satisfy a proxy metric | 
| Bugs & Failures are perceived almost instantly by humans | Failures can be silent and system degradation can be imperceptible |
| Change according to versions | Change is almost a constant in this systems |

## Errors when Testing ML as Regular SW

1. Not analyzing sufficiently how the performance of the model is gonna be quantified and measured (metrics) and its
relation to the context where it's gonna be applied (business part)
2. Test only the model and not the system as a whole (as a consequence of 1.)
3. Missing data testing (also as a consequence of 1.)
4. Rely too much in automated testing and not in production testing
5. Lose sight of the possible divergence of the business measures against the model metrics observed (also as a consequence of 1. and 4.) 

## Meet The Testing Family in ML
  
### Label Tests

- Focus: Labeling system/editor tools/editors
- Goal: Catch poor quality labels or parts of the dataset to avoid model corruption
- Techniques:
  - Get trained labelers
    - We all are prone to biases so...
  - Try to avoid human biases
    - by aggregating labels from multiple labelers
  - Assign labelers a trust score based on how often they are wrong
  - Identify examples in training/testing processes which get different judgement from humans vs computers
    - Relabel those examples again and check if it's true that there were inconsistencies/errors in human and computer judgements
  - Compare older models on new labels
 
### Data Tests

- a.k.a. Expectation Tests
- Focus: Storage and Preprocessing Task
- Goal: Catch bad data or data with quality issues before going to the train pipeline
- Techniques:
  - Define expectations:
    - Assertions for data or
    - rules about properties of each of your data tables at each stage in your data cleaning/preprocessing pipeline
    - e.g. we should expect that Column X in Table A after preprocessing does not contain any null values
    - e.g. we should expect that the average value of Column Y in Table B after preprocessing should be between v1 and v2 
- Tools:
  - [Great Expectations](https://greatexpecations.io)
  
### Infrastructure Test (aka Unit tests in Regular SW development)

- Focus: Training Task
- Goal: Avoid bugs
- Techniques:
  - Use unit test for the parts that are similar to regular sofware (e.g. data preparation and cleaning)
  - Extract a small sample of the dataset to test quickly over it
  - Add tests of a single epoch with the previous extracted small dataset
  - Run frequently
  
### Training Tests

- Focus: Storage and preprocessing Tasks and Training Task
- Goal: Ensure reproducibility of training
- Techniques:
 - Define a set of baseline metrics
 - Pull a fixed dataset representative of the full dataset
 - Check model performance remains consistent against the baseline metrics
 - Consider pulling a sliding window of data
 - As they're slow, run periodically (night or any other specific times)

### Functionality Tests

- Focus: Prediction Task
- Goal: Avoid regressions in the code that makes up your prediction infrastructure
- Techniques:
  - Unit test for the prediction code as for traditional software
  - Load model and test particular examples
  - Run frequently
  
### Evaluation Tests

- Focus: Training Task and Production Task
- Goal: Validate the model to go into production by testing the integration of the two tasks above
- Techniques:
  - Evaluate model in all the important datasets, metrics and slices
    - Traditional metrics (Accuracy, Precision, Recall, etc.) 
    - [Behavioral Tests](https://github.com/francisco-perez-sorrosal/deep-learning-papers/tree/master/Beyond%20Accuracy)
    - Robustness Metrics
      - Related to [Step 5 here](#Errors_when_Testing_ML_as_Regular_SW)
      - I think is very important to asses the overall quality of the model and decide when it's time to react
      - Extract data to understand where the model is gonna perform well or bad
      - Study the feature importance
      - Densitivity to data staleness (vs old data)...
      - ...and data drifts (training vs prod data or different prod data distributions)
        - Important to define metrics for particular data categories (e.g. metrics for the "news" category)
    - Privacy and fairness
      - Will become more important from now on
    - Simulation tests   
      - Understand the interaction of your model with the rest of the environment ("the world")
      - Used in Robotics/Automated Vehicles
      - Hard to model "the world"
    - Shadow tests
      - Testing in production and compare with the results in the offline
      - Detect issues in production
      - Apply the [strangler fig pattern]() to not impact users
  - Compare model against previous one and the baselines
  - This is done only at a particular point in time; so run these tests when you've selected a candidate
  model from the training phase and you want to put it in production.

- Tools: 
  - Robustness: 
    - [what-if-tool](https://pair-code.github.io/what-if-tool/) 
    - [Slice Finder](https://research.google/pubs/pub47966/)

### A/B Tests

- Focus: Serving System
- Goal: Check how the users react to the new model and how the business metrics are affected
- Techniques:
  - Separate a fraction of the request and redirect it to the new model. The rest of the request will
  still targeting the old model which will serve as a control
  - Compare the two cohorts
  - Use monitoring as a tool for evaluation!!!
  - [A/B Testing blog](https://levelup.gitconnected.com/the-engineering-problem-of-a-b-testing-ac1adfd492a8)

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
