# Machine/Deep Learning Topics

## (Artificial) General Inteligence

This has been, is, and will be for some more time at least, the dream of scientist/researchers/engineers in the field of 
artificial intelligence.

In short, achieving a general artificial intelligence assumes the ability of an agent to learn new tasks whilst maintaining 
a general capability on fulfilling previous learned tasks.
This artificial general intelligence requires that the agent does not forget what it has learn -what is called [catastrophic forgetting](#catastrophic_forgetting)
in the literature- and assumes that the agent will continue learning new tasks -called [continual or lifelong learning](#continual-learning).

## Loss Function

Papers:
* [Retrospective Loss: Looking Back to Improve Training of Deep Neural Networks (KDD 2020)](https://arxiv.org/abs/2006.13593)	
Learns from prior training experiences in the form of DNN model states during to guide weight updates and improve DNN 
training performance. The retrospective loss seeks to ensure that the predictions at a particular training step are more 
similar to the ground truth than to the predictions from a previous training step (which has relatively
poorer performance). As training proceeds, minimizing this loss along with the task-specific loss, encourages the 
network parameters to move towards the optimal parameter state by pushing the training into tighter spaces around the 
optimum. Claims implementation is 3 lines of Pytorch code. Interesting paper


## Transfer Learning
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

### Surveys

* [HuggingFace Presentation on Transfer Learning](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit?ts=5c8d09e7#slide=id.g5888218f39_364_0)
* 2010 [A Survey on Transfer Learning](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)
* 2018 [A Survey on Deep Transfer Learning](https://arxiv.org/abs/1808.01974)
* 2020 [A Survey on Transfer Learning in Natural Language Processing](https://arxiv.org/abs/2007.04239)

### Few-shot Learning (Meta-learning)

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

## Calibration

A measure of the confidence of the predictions of a model. Concept comes from weather forecast.

Main Methods:

- Platt Scaling
- Matrix Vector Scaling
- Temperature scaling

Papers:
* [On Calibration of Modern Neural Networks (ICML 2017)](https://arxiv.org/abs/1706.04599)
* [Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration (NeurIPS 2019)](https://papers.nips.cc/paper/9397-beyond-temperature-scaling-obtaining-well-calibrated-multi-class-probabilities-with-dirichlet-calibration.pdf)

<details>
  <summary>Blogs</summary>
 * [How and When to Use a Calibrated Classification Model with scikit-learn](https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/)
 * [Prediction & Calibration Techniques to Optimize Performance of Machine Learning Models](https://towardsdatascience.com/calibration-techniques-of-machine-learning-models-d4f1a9c7a9cf)
 * [Calibration in Machine Learning](https://medium.com/analytics-vidhya/calibration-in-machine-learning-e7972ac93555#:~:text=In%20this%20blog%20we%20will%20learn%20what%20is%20calibration%20and,when%20we%20should%20use%20it.&text=We%20calibrate%20our%20model%20when,output%20given%20by%20a%20system.)
 * [Calibration Tutorial (KDD 2020)](http://kdd2020.nplan.io/presentation) [Github](https://github.com/nplan-io/kdd2020-calibration)
</details>


<details>
  <summary>Videos</summary>
 * [Calibration Tutorial](https://www.youtube.com/watch?v=rhnqZV6eKlg&feature=youtu.be)
</details>



## Causality

* [The Book of Why (Judea Perl)](http://bayes.cs.ucla.edu/WHY/)
* [Causal Inference Intro with Exercises](https://github.com/DataForScience/CausalInference)
* [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)
Learning paradigm to estimate invariant correlations across multiple training distributions. IRM learns a data representation such that the optimal classifier, 
on top of that data representation, matches for all training distributions.

## <a name="continual_learning"></a>Continual Learning and Catastrophic Forgetting

In biology, *Continual Learning* refers to the process of continually gather, update, and transfer skills/knowledge 
throughout life (lifespan).

In ML, it is still a major research problem to solve the fact that neural networks use to catastrophically forget 
previously learned tasks when they are trained in new ones. This fact it is the main obstacle that prevents the 
equivalent of continual learning to be implemented in the field of artificial neural networks.

[This is a summary of the recent (2021) advances in continual learning in NLP.](https://www.aclweb.org/anthology/2020.coling-main.574.pdf)

[Sejnowski](people.md#terrence-sejnowsky) et al. {{ cite tsuda_modeling_2020 }} have developed a NN
 architecture that shows how
 hierarchical gating supports adaptive learning while preserving memories from prior experience. 
They show also how when introducing damages in the model, it recapitulates disorders found on the 
 human Prefrontal Cortex.

### <a name="catastrophic_forgetting"></a>Protocols/Strategies for Solving Catastrophic Forgetting (CF)

One problem with all the different strategies proposed for solving CF is that the field lacks a framework for comparing
the effectiveness of the techniques. This has been addressed by studies like [vandeven2019](refs.md#vandeven2019) and [vandeven2019b](refs.md#vandeven2019b).

The approaches to solve Catastrophic Forgetting can be classified in:

#### Regularization Approaches

https://arxiv.org/pdf/1612.00796.pdf

#### Generative Replay

* [Continual Lifelong Learning with Neural Networks:A Review](refs.md#parisi2020)
* [Brain-inspired replay for continual learning with artiﬁcial neural networks (Nature, 2020)](https://www.nature.com/articles/s41467-020-17866-2.epdf?sharing_token=bkJqxr4qptypBkYehsw_FtRgN0jAjWel9jnR3ZoTv0NoUJpE84DVnSx_jyG1N8KQimOuCCtJtaDabIpjOWE47UccZTsgeeOekV8ng2BR-omuTPXahD4aCOiCIIfIO2IOB-qJOABLKf7BlAYsTBE8rCeZYZcKd0yuWJjlzAEc1G8%3D)
* [Generative replay with feedback connections as a general strategy for continual learning (ICLR2019)](https://arxiv.org/pdf/1809.10635v2.pdf)
* [Three Scenarios for Continual Learning](https://arxiv.org/pdf/1904.07734.pdf)
* [Compositional language continual learning](https://openreview.net/pdf?id=rklnDgHtDS)



# GANs and Creativity


# Research Topics

Unsupervised Learning

Reinforcement Learning
- Unsupervised RL
- Meta-Reinforcement Learning

# Personalization

## Problems


Model <-> User interaction
- In general a model recommends items, and user actions based on those recomendations are used as training data for improving the model.
- Missrepresentation of New items - a group that suffers from algorithmic bias.
- It's interesting to study how recommendation feedback loops disproportionally hurt users with minority preferences
- Features:
  - Empiric (CTR)
  - User history


### Cold start
  - Can be looked as a fairness problems
  - Metrics should be tailored to use cases
  - Fairness methods as a solution.
  - Fairness in new advertisers: measure the severity of the advertiser cold start using fairness metrics and use 
  fairness methods to mitigate it.

Individual fairness -> Similar individuals should be treated similar by the algo
Group fairness -> Individuals from the protected group should get similar treatment as individuals without the protected
attribute. Protected attribute, gender, race...

Methods: Balance for positive class and Calibration

Fairness correction techniques:
- Preprocessing: Correct the training data
- inprocessing: Penalty term to the loss function
- postprocessing: Apply corrections

Address multi-side fairness: Satisfy constraints of all stakeholders (e.g. old/new advertisers). We take the side of the
new items.

Fairness in Ranking

Small changes in scores can lead to large changes in exposure
Static fairness constraints may cause harm in fairness over time
Decomposition of fairness in complex systems: Candidate Generation -> Engagement A model, Engagement B model...
 - Fairness doesn't necessarily decompose
 
 
Faire recommendations with Biased Data (Thorsten Joachims)
History of ranking dates back to the 1960 for finding books in a library
In 1994 with search engines that moved to finding everything
   Maximize the utility of the rankings for the users
In 2020, still we look for Maximize the utility of the rankings. But there are two sides for the utility: 1) for the users (buyers, listeners, readers) but also for the items 2) 
(sellers, artists, writers), etc.! That is the variety of use cases is more diverse

However utility maximization it may not be fair for many candidates, specially if the probability of the top candidates it's very close together
 
Fairness: If two items has similar merits, their exposure should be the same. There are endogenous and exogenous factors:
Fairness of exposure: Endogenous (merit), Exogenous (biases)
 
# Recommendation feedback loops disproportionally hurt users with minority preferences
 - Called "Algorithmic confounding". Perspective
   Users:
   - "I don't get whwat I'm looking for"
   - "This sistem sucks"
   Company:
   - User segmentation
   Technical:
   - Bad training and evaluation protocols
   
Recommendation feedback:
 - Provokes homogeneization of user behaviour
 - Users experience losses in utility
 - Amplifies the impact of the rec system on the distribution of the item consumption
 
Initial data may not be enough 
Poorly tuned models hurt user with minority preferences recommending items further from their preferences
A/B test can weaken overall performance. Too many of them can delay recommending the most relevant items.

 


Matrix factorization
   
  
## Causal modeling applied to messaging at Netflix


 
# Conversational Recommmendation with Natural Language

We want:
- Speaking the user language is essential for a conversational recommendation
- Model that understands natural requests: "I'd like to watch something relaxing"

Soft attribbutes: Property of an item that
 - is not a verifiable fact
 - can be universally agreed on
 - meaningful to compare
 - we can say that one is great than the other
 
Answers to questions
- Polar yes/no question
  - Direct answer
  - Indirect answer (e.g. for being polite). Can include more information. - Do you have kids? - I have 2 daughers
Richness of language

Explainable User Models (Why?)
- What does the system know about me
- How does the system interprets that
- How does it go from user model to recommendations


Observing the language of the user requires new approaches for data collection and models
