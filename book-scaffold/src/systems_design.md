# Designing Machine/Deep Learning Systems

When a business problem involves machine learning and deep learning system, designing such a system to be suitable for production involves much more than just choosing the right model. Several components and/or third party systems will need to be designed, developed and/or integrated in order to ensure a sytem meeting the business expectations and ready to deliver real value.

On one side, careful consideration of the entire training/evaluation/deployment lifecycle will be required, from data collection and preprocessing to model training, deployment, and maintenance.

On the oter side, taking into consideration non-functional features, like ensuring that the system is scalable, secure, and compliant with regulations will be also critical. By taking a holistic approach, you can build a robust, efficient, and effective machine learning system that delivers real value to the business.

So, as we will see, putting a complex system like that into production can be challenging.

In the following sections below, there are described the main considerations when designing an ML system.

Designing a machine learning (ML) system involves a thoughtful balance of various considerations to ensure the system is effective, scalable, and maintainable. Below there is a set of general considerations (described as a sequential pipeline of tasks) when designing an ML system, explained in an educational manner:

## Problem Definition and Scope

### Understanding the Business Objective

Way before diving into the technical aspects, it is critical to understand the problem at hand. The initial question we should ask is, "What is the business need?" Following that, "How will success be measured?", which would allow us determine how to determine the success of the solution. Only once we have a high-level view of the business objective and understand the basic success criteria, can we proceed with more technical aspects such as the selection of data, models, and evaluation metrics.

### Project Scope and Constraints

As with any software project, it is essential to understand the **scope and constraints** of the task at hand. Defining the scope of the problem is critical to avoid setting unrealistic expectations. Also, it is important to identify any constraints, such as time, budget, or computational resources, that could poetentially impact the project's scope. This comprehensive understanding is key to establishing *realistic goals and expectations* for the project.

## Data Management

### Data Collection

Data is critical for (almost) any ML/DL problem. So, **identify the sources  where the data will come from** is one of the first things we'll need to figure out. Do we already have that data, is it generated internally from our current systems?, is it sourced from third-party APIs, or collected from sensors? are the kind of questions we'll have to answer here.

If the data sources are key, its **quality** is even more important. Ensuring the data is of high quality—clean, complete, and relevant will be the next step, as poor data quality will most likely lead to unreliable and useless models.


### Data Preprocessing

If the data is of poor quality we most likely will have to do some **data clieaning**; this involves for example handle missing values, remove duplicates, or correct inconsistencies.

In addition to that we will probably need to do certain **data transformations** to prepare the data as input to the different building blocks of our training/production systems. This encompasses tasks liken normalize/standardize the data, or apply feature engineering to make the data suitable for certain models.

Very close to the tasks above will be the consideration of **data storage** solutions that suit each part of the process best. Choosing appropriately these solutions depending on the data volume and access speed requirements may be influenced by the constrains identified at the begining. Options include shared/cloud fault-tolerant filesystems, data streams, relational databases, NoSQL databases, data lakes, etc.

## Model Selection and Training

The tasks of **choosing an algorithm/model** may come next. The algorithm/model(s) should align with the problem at hand: type—regression, classification, clustering, etc. For example, linear regression for continuous output prediction, or decision trees for classification.

Along with this, a **trade-off between model complexity and interpretability** may be necessary to consider, again, depending on the project constraints. Complex models like the ones involved in deep learning, may offer higher accuracy but may harder to interpret if there's no budget for the adequate tools.

The **training process** shoule be examined next. The **infrastructure** used will need to be considered. Deciding whether to use on-premise servers (if available,) cloud-based solutions, or a hybrid approach for training the model will be put under perspective here. These decisions again will depend on the available computational resources.

**Hyperparameter tuning**, that is, adjust the model’s hyperparameters to optimize performance will have to be considered too at some point. Do we think that the off-the shelf model standard training process will be enough to meet the identified metrics, or we would have to apply techniques like grid search or random search to boost those?

This step will link the training to the **evaluation process**. Here, the **metrics** identified in the first step will be the guiding principle. As we showed, those were selected appropriately based on the problem, but may need to be revisited or complemented here, in case we missed someting. For example, for classification problems, accuracy, precision, recall, and F1-score are commonly used. In regression problems, mean squared error (MSE) or R-squared are more relevant.

Also, in classical ML problems, applying techniques such as **cross-validation** (described in Chapter [Classical ML](classical_ml.md))-e.g. k-fold cross-validation- to ensure that the **model generalizes well to unseen data** may be encouraged. Deep learning-based solutions will have their own protocols, such as proper creation of high quality train/dev/test splits.

## Deployment

The next step in the pipeline turns around thinking the proper infrastructure bells and whistles for deploying the trained model.

### Model Serving

Here, the **serving infrastructure** will need to be evaluated. Decisions on how the model will be served need to be considered here. Options may include REST APIs, batch processing, real-time streaming predictions or ad-hoc approaches, depending on the problem at hand.

Close to the serving infrastructure decision will come questions related to the **scalability of the system**. These questions will have to do to with ensuring that the system can scale to handle increased loads. Techniques like load balancing and horizontal scaling are of essential consideration here.

**Latency and Throughput** may determine also the quality of the final solution for certain problems. **Optimizing** the model and/or certain architectural solutions considering latency requirements —how quickly does the system need to respond? will need to be adressed here. **Caching** mechanisms may help also to reduce load and speed up response times for frequently requested predictions.

### 5. **Monitoring and Maintenance**

- **Performance Monitoring**:
  - **Model Drift**: Regularly monitor the model’s performance to detect any decline due to changes in the input data distribution (model drift).
  - **Logging**: Implement logging to track predictions, errors, and other important metrics. This is vital for debugging and improving the system.
- **Automated Retraining**:
  - **Continuous Learning**: Set up pipelines for automated retraining of the model when new data becomes available, ensuring the model remains accurate over time.
- **Alerts and Notifications**:
  - **Anomaly Detection**: Implement alerts to notify the team of any unusual patterns, like a sudden drop in model accuracy or unexpected input data.

### 6. **Security and Privacy**

- **Data Security**:
  - **Encryption**: Ensure data is encrypted both at rest and in transit to protect it from unauthorized access.
  - **Access Control**: Implement strict access controls to ensure only authorized personnel can access sensitive data and the ML system.
- **Privacy Compliance**:
  - **Regulatory Compliance**: Ensure the system complies with relevant data protection regulations like GDPR or CCPA, especially when dealing with personal data.

### 7. **Collaboration and Documentation**

- **Cross-functional Collaboration**:
  - **Team Involvement**: Involve stakeholders from various departments (e.g., data engineers, software developers, domain experts) to ensure the system is well-integrated and meets business needs.
- **Documentation**:
  - **Comprehensive Documentation**: Document the entire system—data sources, preprocessing steps, model selection rationale, deployment processes, and maintenance plans. Good documentation is crucial for future updates and onboarding new team members.

### 8. **Ethics and Bias**

- **Bias Mitigation**:
  - **Fairness**: Ensure the model is fair and unbiased. Analyze and mitigate any biases that could result from the data or model design.
  - **Transparency**: Be transparent about how decisions are made by the model, especially in high-stakes applications like finance or healthcare.
