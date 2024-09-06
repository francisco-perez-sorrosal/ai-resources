# Designing Machine/Deep Learning Systems

When a business problem involves machine learning and deep learning system, designing such a system to be suitable for production involves much more than just choosing the right model. Several components and/or third party systems will need to be designed, developed and/or integrated in order to ensure a sytem meeting the business expectations and ready to deliver real value.

On one side, careful consideration of the entire training/evaluation/deployment lifecycle will be required, from data collection and preprocessing to model training, deployment, and maintenance.

On the oter side, taking into consideration non-functional features, like ensuring that the system is scalable, secure, and compliant with regulations will be also critical. By taking a holistic approach, you can build a robust, efficient, and effective machine learning system that delivers real value to the business.

So, as we will see below, putting a complex system like that into production can be really challenging and convoluted.

In the sections below, I will outline whay I consider the main topics to address when designing an ML system. These guidelines can also serve as a useful reference for interviews.

Designing a machine learning (ML) system involves a thoughtful balance of various considerations to ensure the system is effective, scalable, and maintainable. Below there is a set of general considerations (described as a sequential pipeline of tasks) when designing an ML system, explained in an educational manner:

## Problem Definition and Scope

### Understanding the Business Objective

Way before diving into the technical aspects, it is critical to understand the problem at hand. The initial question we should ask is, "What is the business need?" Following that, "How will success be measured?", which would allow us determine how to determine the success of the solution. Only once we have a high-level view of the business objective and understand the basic success criteria, can we proceed with more technical aspects such as the selection of data, models, and evaluation metrics.

### Project Scope and Constraints

As with any software project, it is essential to understand the **scope and constraints** of the task at hand. Defining the scope of the problem is critical to avoid setting unrealistic expectations. Also, it is important to identify any constraints, such as time, budget, or computational resources, that could poetentially impact our project's scope. This comprehensive understanding is key to establishing *realistic goals and expectations* for our project.

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

## Monitoring and Maintenance

This involves mainly the following tasks:

### Performance Monitoring

In the Context of ML/DL, performance monitoring refers to the process of continuously tracking and evaluating the performance of the ML model deployed in the system. It involves monitoring various metrics and indicators to ensure that the model is functioning as expected and delivering accurate results. It is an essential non-functional aspect of ML systems design as it is aimed to guarantee that the current model deployed continues to deliver accurate results over time. It enables proactive measures to maintain the model's accuracy, diagnose issues, optimize performance, and enhance the overall system's reliability.

More in detail, this monitoring serves:

#### Model Drift Detection
Identifying model drift is maybe the most important aspect of ML monitoring. A model drift is said to occur when the input data distribution feeding the model changes over time, which may lead to a decline in the model's accuracy. A good monitoring can detect that degradation and raise the alarms or trigger the appropriate actions to maintain the model's accuracy.

#### Logging and Error Tracking
Monitoring has implied recording predictions, errors, and other key performance metrics along the inference pipeline. This information will be critical for diagnosing issues, optimizing performance, and enhancing the overall system's reliability. It will also allow us to track the model's behavior, identify patterns, and troubleshoot any errors or anomalies that may arise during the life of the system.

#### Optimization and Improvement
Overall, by monitoring the performance of the ML system, we will gain insights into its strengths and weaknesses. This information can be used to optimize the model, fine-tune hyperparameters, and improve the overall system's performance. Performance monitoring helps you identify areas where the model may be underperforming or where there is room for improvement.
  
### Continual Learning (Automated Retraining)

This concept is also know continuous learning or lifelong learning; it refers to the capability of a ML/DL model to continually update and adapt over time as it receives new data, without needing to be retrained from scratch. If a process like this is incorporated into our final system, it will ensure that the model can "evolve" and stay resilient to data distribution changes or as new patterns emerge in the real world input data.

This concept is particularly important in dynamic environments where the data is not static, and the model’s utility depends on its ability to keep learning from (aligning to) "new experiences" encoded in the new data distributions coming into the system.

This will be implemented generally by setting up automated retraining pipelines to update the model as new data becomes available. A proactive approach will help maintain the metrics in shape by adapting the model weights to the evolving data patterns. Also during the retraining we need to keep an eye on preserving the past knowledge to avoid the so-called [*catastrophic forgetting*](advanced_ml_dl_topics.md#continual-learning-and-catastrophic-forgetting)

### Alerts and Notifications

Any system is suitable from having sudden drops in model accuracy, unusual input data, or unexpected output patterns, or having some of the resources saturated/underutilized; when this happens, having a good notification subsystem is vital to the life of the system. This will enable our teams to give prompt response to those alerts.

### Security and Privacy

When designing a ML/DL learning system, it is paramount to consider security and privacy non-functional aspects to protect sensitive data and comply with relevant regulations. 

We will have to consider at least:

### Data Security

Incorporating security and privacy considerations into the design of a ML/DL system, will allow us to safeguard sensitive data, protect against unauthorized access, and ensure compliance with applicable regulations/policies. This will help building trust with our users and stakeholders and mitigate potential risks associated with data breaches and/or privacy violations.

#### Access Control
Strict access control policies will ensure that only authorized users can access sensitive data and the system itself. This will help preventing unauthorized access and potential data breaches.

#### Encryption
Implement encryption techniques to protect data both at rest and in transit will complement the point above. This ensures that even if eavesdroppers gain access to the data, they won't be able to decipher it.

### Privacy Compliance

Regulatory Compliance will ensure that the ML/DL system complies with relevant (and relative new since 2018) data protection regulations, such as the General Data Protection Regulation (GDPR) or the California Consumer Privacy Act (CCPA). This is particularly important when dealing with personal data. Compliance with these regulations helps protect individuals' privacy rights and ensures that their data is handled appropriately.


## Collaboration and Documentation

Collaboration and documentation go hand in hand in ML/DL system design and are essential to the success of the final system. Both will ensure that the system is well-integrated, meets business needs, and can be easily maintained and updated.

### Cross-functional Collaboration

As it happens with any complex system, building an ML/DL system is an humoungous effort. So building an effective collaboration process will ensure us that all team members are aligned and working towards a common goal. Involving stakeholders from various departments, such as data curators, data engineers, software developers, and domain experts, will be crucial in our endeavor to designing and implementing a successful system. Each team member will bring unique expertise and perspectives, contributing to the overall effectiveness and efficiency of the final system. Collaborating across functions helps ensure that the system aligns with business objectives and addresses the needs of different stakeholders.

### Documentation
  
Having a comprehensive knowledge base of documentation of the entire ML/DL system is essential for future updates, maintenance, or onboarding our new new team members. It should cover various aspects, including data sources, preprocessing steps, model selection rationale, deployment processes, and maintenance plans. With these details documented, the team can easily refer back to them, understand the system's design choices, and make more informed decisions. 

Comprehensive documentation also facilitates knowledge sharing and collaboration among team members, enabling smoother transitions and reducing the risk of knowledge gaps.

## Ethics and Bias

In the context of ML/DL system design, it is crucial to address ethics and bias considerations.
By addressing ethics and bias, we will ensure that our system is fair, unbiased, and accountable. It will prevent discriminatory outcomes and will promote trust and confidence in the outputs delivered by our system. Incorporating, fairness and transparency in our design process will ground and align us more with the ethical considerations our users and stakeholders may have.


Some concepts to consider are related to **Bias Mitigation**. "Bias" in this context refers to systematic errors or prejudices that can affect the performance, fairness, and outcomes of a ML/DL system.

### Fairness

When designing such a system, it is important to ensure fairness. Fairness in this context means analyzing and mitigating any biases that could result from the data or model design. These biases can arise from various sources, such as the training data itself or other biased features used in the model.

The main forms in which biases can manifest are:

#### Data Bias
The training data used to develop our system does not represent the real-world population or contains inherent biases.

#### Algorithmic
Our models produce results that are systematically prejudiced, most likely due to erroneous assumptions in the machine learning system design process

#### User Bias
The final design of a system reflects the biases either of its designers or of their users, potentially leading to exclusion or unfair treatment of certain groups.

So it will be essential to carefully examine our data and final model to identify and address  potential biases to ensure fair and equitable outcomes.

### Transparency

Transparency is the other important aspect for addressing ethics and bias considerations in our ML system design. 

Being transparent about how decisions are made by the model, especially in high-stakes applications like finance or healthcare will build trust with our users. 

In the same way, providing explanations for our model's predictions or decisions can help building trust and understanding among our users and stakeholders. 

Finally, transparency can also help us identifying and rectifying any biases or unfairness in the system.

## Resources

### Books

#### Overall System Design

[Acing the System Design Interview, Zhiyong Tan](https://www.manning.com/books/acing-the-system-design-interview)

[System Design Interview – An insider's guide, Alex Xu](https://www.amazon.com/System-Design-Interview-insiders-Second/dp/B08CMF2CQF)

[System Design Interview – An insider's guide Vol.2, Alex Xu](https://www.amazon.com/System-Design-Interview-Insiders-Guide/dp/1736049119)


#### Machine Learning System Design

[Designing Machine Learning Systems, Chip Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
- [DMLS Booklet](https://github.com/chiphuyen/machine-learning-systems-design)

[Machine Learning System Design Interview, Ali Aminian, Alex Xu](https://www.amazon.com/Machine-Learning-System-Design-Interview/dp/1736049127)

[AI Engineering, Chip Huyen](https://www.oreilly.com/library/view/ai-engineering/9781098166298/)

### Github Resources

[Interview Questions](https://github.com/alirezadir/Machine-Learning-Interviews/tree/main/src/MLSD)

