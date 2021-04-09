# Aproaches and Flavors in AI/ML

## Approaches to ML

The following categorization is done for simplification and is based on the one done by Pedro Domingos in 
{{ #cite domingos_master_2015 }}. The influence of each one of them has varied along the evolution of machine learning, 
but in the end all them influence each other in some way or another; so, many times, the "boundaries" we humans
tend to trace among them (or in any other categorization) they only exist in our imagination.

### Symbolist

Learning is viewed as a kind of inverse approach of deduction. Very influenced by logic and the need of humans
to try to represent abstract problems in a human-readable way to facilitate communication with each other. It had early 
successes in early AI computer programs, in particular in the era of [expert systems](vocabulary.md#Expert System).

### Connectionist

Basically this approach is based on studying the recent advances in neuroscience related to how the brain works 
and try to do a reverse engineering process of what has been learn. As neuroscience is indeed influenced by physics
the connectionists model also the neurons based on the 

#### From the Early Days in Connectionism, towards Deep Learning

```textmate
Alan Turing's Intelligence Machinery (1948) -> Perceptron (1962) -> Hopfield Net (1982) -> Bolzman Machines(1985) -> Backprop (1986) -> First NIPS Conf. (1987) -> Deep Learning matures as a field on its own in NIPS Conf. (2012)
```


### Evolutionary
 
Influenced by genetics and evolutionary biology, this approach tries to simulate this evolutionary environment
through computers. 

### Bayesian
 
This approach is based mainly in applying probabilistic inference to mimic the learning process and of course is
strongly influenced by mathematics and statistics. 

[Causal inference](topics.md#Causality) can also be included in this approach. Human beings we tend to approach situations in terms of
cause and effect; this framework of thinking often modifies our behaviour when we find (or at least we think we've 
found) WHY a certain event has occurred and what are its consequences.

Some of the main advocates of this approach in its modern conception is [Judea Pearl](people.md#Judea_Pearl).

#### Analogizers

They consider that learning is done by abstracting and reconciling experiences. Influenced by psychology and cognitive
sciences in general and math optimization.

### Actioninst

The approach is based on the fact that a limited set of basic behaviours or states in a system can lead to more complex 
ones, by correcting themselves through observation and subsequent action. This has to do to what is known as [reinforcement learning](#Reinforcement Learning)
and [behavioral robotics](https://en.wikipedia.org/wiki/Behavior-based_robotics). 
In the case of robots, this paradigm advocates that the most effective mechanism to learn sensory-motor cognitive 
abilities like walking, should be based on interactions with the environment. After all, this is more or less how toddlers
learn to walk; model these kind of interactions programmatically through abstract reasoning based on rules, 
it's inefficient and prone to errors. 
 
### Autonomic/Survivalistic

The approach is based on the fact that intelligent complex systems like humans, are able to survive on their own. In
order to do that, we rely in parts of the brain that either are not well understood now or they haven't taken into account
yet in the artificial systems implemented. This can be related to the field of [autonomic computing](https://en.wikipedia.org/wiki/Autonomic_computing).

Some of these brain areas are:

- Hypothalamus, which controls basic non-conscious subsystems related to maintaining body temperature, thirst, hunger 
and other homeostatic systems. Some of these could be mimicked by a computer system; e.g. the case of feeding, an 
intelligent ingestion data pipeline could be feeding a self-adaptive a neural network.

- Cerabellum, which is located near the brainstem, and is responsible to coordinate movements. In the case of robotics,
this is related to the field of [behavioral robotics](https://en.wikipedia.org/wiki/Behavior-based_robotics) and all the
advances done by companies such as [Boston Dynamics](https://www.bostondynamics.com/), Samsung STAR Labs or NASA.

### Adaptable

Kind of similar to the [Autonomic/Survivalistic](approaches.md#Autonomic/Survivalistic) approach but taken from the 
point of view of computer science instead of the survival of species.
When modeling learning in with an artificial machine/algorithms, why be limited by Turing Machines? Human computational
power is way beyond what a Turing machine is capable to do and this approach tries to mimic that. The adaptable approach
is related to what is known as the Super-Turing Computation or [Lifelong Learning](topics.md#Continual Learning and Catastrophic Forgetting). 
This is also related to the [Time-Aware Machine Intelligence (TAMI) program](https://www.darpa.mil/program/time-aware-machine-intelligence) at DARPA, 
which aims to model meta-learning.
  

## Flavors in ML

### Reinforcement Learning

A branch of machine learning that was influenced by the associative learning approach observed in animals and
newborns/young adults.