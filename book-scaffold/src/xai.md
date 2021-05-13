# Explainability

Using machine learning models as a black box is the default approach used in many modern AI-based applications. These apps
are used most of the time in non-critical domains. However, many other domains and applications require more than just 
a working agent or ML model. "Trust" is the keyword in this context. Can we trust this ML model? Can we be sure that 
a robot will react reliably and appropriately to a particular human interaction? In order to understand
that, we need to know what happens in that black box. This is essentially the goal of Explainable Machine Learning (XAI). 

XAI aims to complement the model AI ecosystem with new tools and techniques that will allow AI users to understand, 
control and ultimately trust any ML model or agent.
 
The essential part of a XAL intelligent agent will encompass what is called [Interpretability/Explainability](vocabulary.md#Interpretability)
and [Explanation](vocabulary.md#Explanation) {{ cite miller_explanation_2018 }}.


## Model Interpretability
Intelligibility and explanation are critical in many domains (health, crime prediction...) [Blog entry](https://medium.com/analytics-vidhya/model-interpretation-with-microsofts-interpret-ml-85aa0ad697ae)

Models as black box methods: Shap, LIME
Glass box models: Explainable Boosting Machine (EBMs) are the SotA

Papers:

* Neural Additive Models: Interpretable Machine Learning with Neural Nets {{ cite agarwal_neural_2020 }}
* A Unified Approach to Interpreting Model Predictions (NIPS 2017) {{ cite lundberg_unified_2020 }}

Books:
* [Interpreable Machine Learning Book A Guide for Making Black Box Models Explainable (Christoph Molnar, 2020)](https://christophm.github.io/interpretable-ml-book/)

Tools:
* [Lime](https://github.com/marcotcr/lime)
* [Shap](https://github.com/slundberg/shap)
* [Interpret ML](https://github.com/interpretml/interpret) Reference impl of EBMs
* [Shap Tutorial (Includes BERT examples)](https://nbviewer.jupyter.org/github/slundberg/shap/blob/master/notebooks/general/Explainable%20AI%20with%20Shapley%20Values.ipynb)
* [Captum](https://captum.ai/) For Pytorch
* [Explainable AI works by Jesse Vig (PARC Researcher)](https://jessevig.com/)
* [Testing and Explainability (Lecture 10 in FullStackDeepLearning)](https://fullstackdeeplearning.com/spring2021/lecture-10/)
