# Generative AI

## Origins of Generative AI

We can say that Generative AI has been an active area of research since -at least- the decade of 1960s, although it wasn't referred like that. During the period of 1964 to 1967, J. Weizenbaum developed what we can consider one of the first expert systems with the form of a chatbot, [ELIZA](origins.md#expert-systems) {@@cite weizenbaumELIZAComputerProgram1966}. At that time, it looked like computers where going to take over the work of many experts; the posibilities seemed endless. However the optimism came to an end with the [AI Winter of the 70s](origins.md#ai_winter_70s).

If we fast forward in time to the last few-years, the now called Generative AI -or just GenAI- has revolutionized (again?) the field of artificial intelligence. The generative models enable machines to create content that is becoming indistinguishable from human-generated content. Although not completely fair, the recent origins of this transformative technology can be traced back to the development of the Generative Pre-trained Transformer (GPT) models and their parent company, OpenAI.

We can put the first significant milestone in the release of the original GPT paper in 2018 {@@cite radfordImprovingLanguageUnderstanding}. Originally entitled "Improving Language Understanding by Generative Pre-Training", the model presented in the paper demonstrated the power of pre-training a [transformer architecture](transformers.md) with a large corpus of text data and then subsequently fine-tuning it for specific tasks. GPT  followed the original transformer work {@@cite vaswani_attention_2017}, having a 12-layer decoder-only transformer blocks with masked self-attention heads with 768 dimensional states and 12 attention heads. The total number of parameter was 117 million, which was a significant achievement at the time (although BERT large had 340M). This contrasts the bi-directional encoder-decoder transformer architecture of [BERT](nlp.md#bert) (also published in 2018) {@@cite devlin}.

The success of the original GPT model just set the groundwork for more advanced models in the forthcoming years. OpenAI introduced GPT-2 in 2019 {@@cite radfordLanguageModelsAre}. First, GPT-2 showcased a significant leap in scale, getting 1.5 billion parameters. But also in performance, as this model was capable of generating coherent and contextually relevant text, pushing the boundaries of what was possible with GenAI.

GPT-3 was released the following year, 2020; and it marked another significant milestone in the evolution of generative models. Again, it's an order of magnitude larger than its predecessor, reaching the ashtonishing figure at that time of 175 billion parameters. The massive increase in scale allowed GPT-3 to generate even more coherent and contextually accurate text, making it one of the most powerful language models ever created.

Architecturally, GPT-3 was not very surprinsing; it followed the same transformer-based design as its predecessors. However, it benefited from an extensive training process on a selected and diverse set of datasets. This enabled GPT-3 to perform a wide range of NLP tasks this time with minimal fine-tuning, including translation, question-answering, and creative writing.

The release of GPT-3 beyond demonstrating the potential of large-scale generative models, started sparking discussions about the ethical implications and societal impact of this new breed of powerful AI technologies.

But maybe was on November 30, 2022 when the true inflection point for GenAI became aparent; In those days, during the annual Neurips conference, OpenAI released the so-called ChatGPT. This chatbot -like many others before- was an experimental AI model, but quickly became viral due to the "sensation" it created for the users of "being truly chatting with some other persona" on the other side.

According to OpenAI, ChatGPT was fine-tuned from GPT-3.5 series models on an Azure AI supercomputing infrastructure. It re-introduced the so-called Reinforcement Learning from Human Feedback (RLHF) technique, presented in the paper InstructGPT {@@cite ouyangTrainingLanguageModels2022} in January of 2022, with minor differences in the data collection process.

The human-supervised fine-tuning task, means that humans "trained an AI" by providing examples of dialogues/conversations playing the two roles: a human and an AI. These humans were given specific training to help them with that task. These dialogues were mized with the transformation of the original InstructGPT dataset into a dialogue format.

The reinforcement learning (RL) part comes into play when a reward model for reinforcement learning has to decide which one of two or more responses is better using a technique called Proximal Policy Optimization (PPO) during several itereations. In order to do that, a comparison dataset had to be collected/created, rainking 2+ responses by quality. To do so, from the human-AI dialogues mentioned above, model-written responses where randomly selected and some completions were then subsequently sampled, having the AI trainers to rank them.

All these pioneering efforts have paved the way for subsequent advancements in the field, leading to the development of even more sophisticated models that continue to shape the future of AI-driven content generation.

## Reinforcement Learning from Human Feedback (RLHF)

![RLHF](images/rlhf.png)

The RLHF Process described in the InstructGPT paper and in the figure above consist of:

### Data Collection

1. Hire 40 contractors to label data based on their performance on a screening test.
2. Collect a dataset of human-written demonstrations of desired output behavior on prompts.
3. Collect a dataset of human-labeled comparisons between model outputs on a larger set of API prompts.

### Supervised Learning Baseline

Train supervised learning baselines using the collected human-written demonstrations.

### Reward Model (RM) Training

Train a reward model on the dataset of human-labeled comparisons to predict preferred model outputs.

### Fine-Tuning with PPO

1. Use the reward model as a reward function.
2. Fine-tune the supervised learning baseline to maximize the reward using the PPO algorithm.

### Evaluation

1. Evaluate models by having labelers rate the quality of model outputs on a test set.
2. Conduct automatic evaluations on public NLP datasets.
3. Train models of different sizes (1.3B, 6B, and 175B parameters) using the GPT-3 architecture.

### PPO

### DPO

## New Architectures Beyond the Transformer