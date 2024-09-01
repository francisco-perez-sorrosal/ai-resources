# "Classical" Machine Learning

## Overview of Machine Learning

Machine learning (ML) has been traditionally the field of artificial intelligence (AI) that enables systems to learn from data and make decisions with minimal human intervention. Unlike traditional programming, where rules are explicitly programmed, ML models identify patterns in data to make predictions or decisions.

In the recent years, the term *"Classical Machine Learning"* has been coined to refer and encompass the initial set of algorithms and techniques that, till the begininig of thew century, have been foundational in the field of machine learning for decades. These methods typically include:

- linear regression
- logistic regression
- decision trees
- support vector machines (SVM)
- k-nearest neighbors (KNN)

Most of these classical ML methods often require the so-called feature engineering, where domain knowledge is used to create meaningful input features from raw data. These models are generally effective for structured data, that is, those where relationships between input features and output labels are relatively straightforward.

As of 2024, the now so-called Classical Machine Learning serve as the foundational bedrock of more modern ML techniques. The algorithm and methods in this context still provide very robust and interpretable models supporting a wide range of applications. As we will see later, Deep Learning (DL) partly builds on these foundational techniques, and has enabled more recently new breakthroughs in fields that require processing even more complex, high-dimensional data. Finally, the term Generative AI has been introduced in the last few years to take DL methods a step further, allowing machines not just to learn from data, but to create new content that mimics or enhances human creativity. As we will see, each of these areas has its unique strengths, limitations, and use cases. Toghether, they form the broad and rapidly evolving landscape of the modern artificial intelligence.

### Types of Machine Learning

TODO Add the Supervised vs Unsupervised vs Semi-Sup

## Key Algorithms in Classical Machine Learning

TODO Complement with the new stuff

- **Linear Regression**:
  - A method used for predicting a continuous dependent variable based on one or more independent variables. The model assumes a linear relationship between the input variables and the output.
  - **Formula**: \( y = \beta_0 + \beta_1x_1 + \ldots + \beta_nx_n + \epsilon \)
  - **Applications**: Predicting house prices, stock market trends, and other scenarios where the relationship between variables is assumed to be linear.

- **Logistic Regression**:
  - A classification algorithm used for binary classification problems (i.e., where the output can be one of two classes). It predicts the probability that a given input belongs to a certain class.
  - **Formula**: \( \text{logit}(p) = \log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1x_1 + \ldots + \beta_nx_n \)
  - **Applications**: Spam detection, disease diagnosis, and credit scoring.

- **Decision Trees**:
  - A non-parametric supervised learning algorithm used for classification and regression. The model splits the data into subsets based on the value of input features, creating a tree-like structure.
  - **Key Concepts**: Nodes, branches, and leaves. Each internal node represents a decision based on a feature, each branch represents the outcome of the decision, and each leaf node represents the final prediction.
  - **Applications**: Customer segmentation, fraud detection, and decision-making processes.

- **Support Vector Machines (SVM)**:
  - A supervised learning model that finds the optimal hyperplane that best separates the data into classes in a high-dimensional space. SVM can be used for both classification and regression tasks.
  - **Key Concepts**: Hyperplane, support vectors, margin. SVM aims to maximize the margin between the data points of different classes.
  - **Applications**: Image classification, text categorization, and bioinformatics.

- **K-Nearest Neighbors (KNN)**:
  - A simple, non-parametric algorithm used for classification and regression. The model classifies a data point based on the majority class among its k nearest neighbors in the feature space.
  - **Key Concepts**: Distance metrics (e.g., Euclidean, Manhattan), choosing k (number of neighbors).
  - **Applications**: Handwritten digit recognition, recommendation systems, and medical diagnosis.

- **Naive Bayes**:
  - A probabilistic classifier based on Bayes' theorem, assuming independence between the features. Despite the naive assumption of independence, it performs well in many real-world applications.
  - **Formula**: \( P(C|X) = \frac{P(X|C)P(C)}{P(X)} \)
  - **Applications**: Spam filtering, sentiment analysis, and document classification.

## Model Evaluation and Validation

TODO Complement with the new stuff

- **Train-Test Split**: The dataset is divided into a training set (used to train the model) and a test set (used to evaluate the model's performance).
- **Cross-Validation**: A technique where the dataset is split into multiple subsets, and the model is trained and evaluated multiple times, each time using a different subset as the validation set. This helps in getting a more reliable estimate of model performance.
- **Performance Metrics**:
  - **Accuracy**: The ratio of correctly predicted instances to the total instances.
  - **Precision, Recall, and F1-Score**: Metrics particularly important for imbalanced datasets.
  - **Confusion Matrix**: A matrix that shows the actual versus predicted classifications, helping to identify true positives, false positives, false negatives, and true negatives.

## Regularization Techniques

TODO Complement with the new stuff

- **Purpose**: Regularization techniques are used to prevent overfitting by adding a penalty to the loss function for more complex models.
- **Types**:
  - **L1 Regularization (Lasso)**: Adds the absolute value of coefficients as a penalty to the loss function. Encourages sparsity in the model (some coefficients become exactly zero).
  - **L2 Regularization (Ridge)**: Adds the square of the coefficients as a penalty to the loss function. Encourages smaller coefficients, but all coefficients remain in the model.
  - **Elastic Net**: A combination of L1 and L2 regularization, balancing sparsity and small coefficients.

## Feature Engineering

TODO Complement with the new stuff

- **Feature Selection**: The process of selecting the most relevant features for the model to improve accuracy and reduce overfitting.
- **Feature Scaling**: Techniques like normalization and standardization to ensure that all features contribute equally to the model, especially important for algorithms like SVM and KNN.
- **Feature Transformation**: Applying mathematical transformations to features, such as logarithms or polynomial expansions, to capture nonlinear relationships.

## Summary and Next Steps

Machine learning is a vast and continuously evolving field getting feedback from many different areas, so it's important to keep learning and exploring new concepts and techniques. With new algorithms, techniques, and applications emerging regularly, staying up-to-data with the latest research papers, attend conferences and webinars, and follow reputable blogs and forums will keep yourself informed about the latest advancements in the field.

However, as we showed in this chapter a strong grasp of the fundamentals of the so-called "classical machine learning" algorithms, model evaluation techniques, and feature engineering is crucial not only for understanding and building new and robust machine learning models that conform (or will conform) many modern (future) software applications, but to continue this learning process.

With a good understanding of the foundations, the next steps will be exploring more recent advanced ML Techniques: For example, dive deeper into more complex algorithms such as ensemble methods (e.g., random forests, gradient boosting), support vector machines, and neural networks. These techniques can provide more powerful models for complex and high-dimensional data.

Also gaining practical hands-on experience is crucial. Applying the acquired knowledge by working on real-world ML projects, finding datasets and problem statements that interest you, and start building and evaluating machine learning models will be very helpful to solidify your understanding and develop practical skills.
