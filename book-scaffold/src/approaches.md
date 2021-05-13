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

Basically this approach is based on studying the recent advances in cognitive sciences related to how the brain works 
and try to do a reverse engineering process of what has been learn. In cognitive sciences there have been and still
 are many  researchers that advocate for ToMs that posits that the mind is only made up of randomly arranged neurons. We 
can say Artificial Neural Networks (ANNs) were the main result of this line of though.
 
However, this view is a very restricted one for many other researchers, for example [Gary Marcus](people.md#gary-marcus) 
There are other trends like Neural-Symbolic systems that try to combine ANN and logic
 {{ cite besold_neural-symbolic_2018 }} to try to build systems able to learn and reason.

As neuroscience is indeed influenced by physics
the connectionists, may model also the artificial neurons based on the physical properties observed in real neurons
. That's why in this approach we can include also the neuromorphic approach, which models neurons in a more similar
 fashion to the neurons in the brain. For example, a neuromorphic neuron uses  just 1-bit spike to communicate with other neurons. 
Neurons can send these spikes in order to activate more or less neurons connected to them. [Carver Mead](people.md#carver-mead)
was one of the initial contributors in developing this approach. More recently, [IBM](https://www.ibm.com/blogs/research/category/neuromorphic-computing/?mhsrc=ibmsearch_a&mhq=neuromorphic%20computing) 
or the [e-lab](http://e-lab.github.io/index.html) at Purdue university have continued developing these ideas of this
 field, now called [neuromorphic computing](vocabulary.md#neuromorphic-computing). Although there are [simulators
 ](http://apt.cs.manchester.ac.uk/projects/SpiNNaker/) 
in conventional hardware for modelling this approach, this is highly ineficient, as for example, among other limitations
, 32-bits are used to model 1-bit spikes, so in order to fully develop neurorphic computing special hardware is needed. 

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