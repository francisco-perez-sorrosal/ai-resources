# Agents

In the context of ML/AI, an agent represents an entity/system that is able to perceive its environment through sensors (input), processes the information perceived, and then acts upon the environment through actuators (output) to achieve a certain goal. The agent performs this input/output loop in an autonomous/semi-autonomous way.

## Main Features

0. *Goal-Oriented* Agents operate with a goal or objective to accomplish, whether that is maximizing a reward in RL, solving a problem, or following a set of rules.
1. *Perception* Agents receive information from their environment; this could be through raw data, such as images, text, or sensor readings.
2. *Processing* Agents process the information they perceive from the environment to make decisions (e.g. take an action); this could involve algorithms, reasoning, learning, or planning to determine the next best action.
3. *Action* Agents act on the environment to achieve the specific goal they have; depending on the type of agent, actions can be physical (e.g. a robotic arm moving its fingers) or virtual (e.g. a software agent making a sending an email).

## Types

There are many classification of agents. For example:	

* *Reactivity vs. Deliberation* Does the agent react immediately to stimuli, or does it think ahead and plan?
* *Internal Representation* Does the agent have a model of the environment to track past states or predict future states?
* *Goal-driven* Is the agent’s behavior directed toward achieving specific goals?
* *Learning and Adaptation* Does the agent learn and improve its actions over time?

The following one is based on the level of complexity and decision-making capabilities of the agents, and more specifically considers how an agent perceives its environment, processes information, and selects actions to achieve its goals:

1. Reactive Agents
Agents that act solely based on their current perception of the environment, without storing any memory or history of previous states. They are limited to their current percepts and can not “think ahead”. A robot that reacts to obstacles by turning away when it detects something in its path.
2. Model-Based Agents
Agents that maintain an internal representation (or model) of the environment. They use this model to keep track of past states and plan future actions by understanding how their actions affected the environment so they can plan the new ones better over time. A chess-playing AI that keeps track of the game board and plans its moves by simulating possible outcomes.
3. Goal-Based Agents
Agents that have a specific goal, and their decisions are driven by the desire to achieve that goal. They might consider future consequences and plan actions accordingly. A self-driving car that plans its route to reach a destination while avoiding traffic and accidents.
4. Utility-Based Agents
Agents that that aim at maximizing a utility function, which represents their degree of satisfaction or success. So, they not only aim to achieve a goal but also evaluate the best way to reach it, considering factors like efficiency, speed, or safety. An AI in a recommendation system (like for movies or shopping) that tries to maximize user satisfaction by suggesting the most relevant content.
5. Learning Agents
Agents that can improve their performance over time by learning from past experiences and adjusting their behavior. They might use techniques like reinforcement learning or supervised learning. A robot that learns to navigate a maze by trying different routes and learning from mistakes to find the shortest path.

So, based on its perception and memory capabilities, agents can be reactive or model-based. Based on their goal orientation, utility-based or goal-based. Based on their decision-making strategy agents can be reactive, goal or utility-based. And based on their planning and future considerations abilities, model-based agents and goal-based agents can consider the future consequences of their actions (they plan ahead and reason about the world), while reactive agents do not engage in forward-thinking or planning.

## Reasoning

For human beings, reasoning is considered as the cognitive process that aims at drawing conclusions, making decisions, or solving problems based on the use of available information, logic, and thought. The process usually requires a deliberate mental activity of analyzing, synthesizing, and evaluating information to form judgments or derive new knowledge. A parallel concept could be what D. Kahneman and A. Tversky describe as "System-2" Thinking.

The reasoning process may follow different pathways, including *deduction* (from a general principle/premise, apply it to specific instances to derive a certain conclusion, e.g. “All humans are mortal. Socrates is a human. Therefore, Socrates is mortal.”), induction (making generalizations based on specific observations, e.g. “I’ve seen many swans, and they are all white. Therefore, all swans must be white.”), *abduction* (construct a hypothesis -or best-guess explanation- based on incomplete information, e.g. if we live alone, get home, find the door open, and our stuff is moved, we might reason that someone broke in, even without seeing the thieves), and *analogy* (make a comparison between two scenarios or concepts and conclude that, due to the fact that they are similar in some aspects, we can infer that they might be similar in others, e.g. when we say “Life is like a journey”, we are conceptualizing life as a path with challenges, decisions, and growth, much like traveling to an unknown territory). All these pathways are mimicked in some way or another in modern ML/AI agentic approaches.

The pioneering attempts in the field of ML/AI about reasoning laid the foundation for modern AI systems. They primarily involved symbolic reasoning, logic, and knowledge-based approaches. Some of them were:

1) The Logic Theorist (1956) Program – Allen Newell and Herbert A. Simon; Ofter considered the first AI program, it introduced for the first time a symbolic approach to reasoning, in this case over mathematical theorems included in Newton's *Principia Mathematica* . The program could manipulate abstract symbols (that is, logic statements) to derive conclusions, an early form of automated theorem proving.

2) General Problem Solver (1957) – Allen Newell and Herbert A. Simon; This attempt to do a generalist reasoning program, laid the groundwork for future research into automated reasoning and problem-solving, some of which we're experiencing as we speak these days. It introduced the idea of breaking down complex problems into smaller subproblems, which now we can see also in agentic systems and [CoT prompting](vocabulary.md#chain-of-though-cot), and using means-ends analysis (reduce the gap between the current agent's state and the goal state by identifying actions -*means*- that will bring the agent closer to get the desired outcome, or *ends*) to find solutions.

3) Perceptron (1958) – Frank Rosenblatt; Considered one of the first neural network models, it was designed to recognize patterns and classify objects based on their input features. It diverged from the original symbolic-based approaches, proving that a neural network could “learn” from data and generalize beyond specific examples. This has been proven crucial for later developments in reasoning based on pattern recognition and statistical learning in the 90s and specially in the 21st century.

4) Others early attemps of reasoning, include DENDRAL (Edward Feigenbaum, Bruce Buchanan, and Joshua Lederberg in 1965), SHRDLU (Terry Winograd, 1970), MYCIN (Edward Shortliffe, 1972), Frames and Scripts (Marvin Minsky and Roger Schank, 1970s), Prolog (Alain Colmerauer and Robert Kowalski, 1972) and SOAR (Allen Newell, John Laird, and Paul Rosenbloom, 1980s).

## RL Agents


## LLM Agents

Since the advent of GenAI, there's been a massive interest on LLM-base agents. These agents are systems that leverage a LLM to interpret instructions, perform tasks, and interact with different environments through tools (a.k.a. function/API calls) based on the natural language input provided by the users.

A major limitation of LLMs is related to the so-called common reasoning. They are not conscious beings and they struggle correcting their own reasoning errors without getting some external feedback.

Presenting an LLM with step-by-step prompts, the so-called Chain-of-Though (CoT), to guide the reasoning process helps achieving better performance on solving complex reasoning tasks. The least-to-most prompting technique in LLMs teach the model to break a complex task into a sequence of simpler and smaller steps. This goes hand in hand with "self-consistency” techniques, which require the model to produce multiple solutions for finally selecting the most consistent answer. Also, as human-beings presenting them information in an illogical order makes them decrease the performance on the task at hand.

### Memory

Long-term memory agents enables agents to retain information across sessions (using the LLM list of past messages), while short-term memory does not persist over new tasks.

### ReAct

ReAct {@@cite react} describes a pattern of agent interaction with its environment that basically consists on reasoning first on a particular task to further take an action on the environment with the aim of accoplishing the task at hand. In dynamic environments, the interaction can be seen as a continuous virtuous cycle in which the reasoning part informs the particular selection of an action, and the execution of action (seen as an observation over the environment) informs further the reasoning process.

A limitation of LLM Agents that do not incorporate real-time feedback from their actions is that they may struggle in adapting to dynamic content and user interactions.

### Retrieval-Augmented Generation


## Multi-Agent Systems

In more complex scenarios, multiple agents can interact within the same environment, either cooperating or competing. These multi-agent systems can be used in areas like simulation of economies, swarm robotics, or coordination in distributed AI systems.
