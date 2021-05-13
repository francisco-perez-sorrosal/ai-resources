# Datasets

One of the most important elements in any machine learning process are the datasets. Initially, they contain the raw data
that will fed ML models or any other intelligent agent. However, most of the time, that raw data it's not used directly
to feed a model. Usually, the raw data is "polished" in a pre-processing step 
in order to homogenize, transform, curate, complete, and or extract relevant pieces of information or [features](vocabulary.md#features)
before it is used as an input to a model.  

The main datasets involved in any ML process for creating a model are:

- **Train**, which is the data your model will train on
- **Dev**, which is a representation of the test dataset used to do hyperparameter tuning, select features or make any
 other
decision over the model. In scikit-learn use to be called **hold-out cross-validation set**. 
- **Test**, which is used exclusively for evaluating the metrics selected for the model. It should not be used to make
 any other decision about the process of developing the final model.