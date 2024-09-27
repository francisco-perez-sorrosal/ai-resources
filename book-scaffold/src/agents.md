# Agents


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

