# NLP

## Frameworks

* [Huggingface](https://huggingface.co/) The de-facto standard framework for modern NLP.
* [X-Transformers](https://github.com/lucidrains/x-transformers) A new repo, implementing also the later
 advances in the spectrum of Transformer-based models.
* [NLP Architect](https://github.com/NervanaSystems/nlp-architect)

# Blogs & Repos

* [NLP Bootcamp](https://github.com/neubig/lowresource-nlp-bootcamp-2020) CMU lectures on NLP by visitors to the
 Language Technologies Institute.


# NLP Topics

## Text Categorization

One of the classical problems in NLP.

**Goal**: Assign labels/tags to text examples (e.g. sentences, paragraphs, documents...)
**Options for doing text annotation**:
- Manual - Reliess on humnans; Because of that fact this approachss doesn't scale, is costly, and error prone.
- Automatic - The current trend due to the increasingly amount of text examples required for many applications in the
industry.
  - Rule-based methods
    - Use a set of predefined rules
    - Require domain knowledge from experts
  - ML-driven methods
    - Use a set of prelabeled examples to train models.
    - Learn -during a training phase- based on observations of data contrasted against the true/gold labels already
    tagged by domain experts for a certain number of the so-called train examples.
    - The final model obtained with this method has learned associations between the text and the labels

We will focus on this approach only, and mainly on the ML-driven methods.

__Applications__:
1. Sentiment analysis
2. News classification
3. Content moderation
4. Spam filtering
5. Question-answering
6. Natural language inference
...

### Procedure

The traditional way of doing text classification consists of these steps:

0. **Dataset creation** - Create (or download, if a well-know industry used dataset is considered to be used) at least
 two datasets from the text examples available: train and test. See [Datasets](datasets.md) section for more information.
1. **Preprocessing** - Some handcrafted [features](vocabulary.md#feature) are [extracted](vocabulary.md#feature
-engineering) from the train and test datasets. This may require also to do some transformations on the raw input data.
2. **Training** - From each train example, use the features extracted + its associated label, as input to train a model
 that will
learn associations from the features and the labels to make predictions on new input features.
3. **Testing** - Feed a model with the features extracted from each test example to the train model to obtain a
 prediction.
4. **Evaluation** - Take each prediction obtained and contrast it with the corresponding true/gold label for test
 examples and calculate the required [metrics](metrics.md) for the classification problem at hand.

Popular Algorithms used for text classification are [Naive Bayes](algorithms_and_model_architectures.md#naive-bayes),
[SVMs](algorithms_and_model_architectures.md#support-vector-machines), [HMMs](algorithms_and_model_architectures.md#hidden-markov-models),
[GBTs](algorithms_and_model_architectures.md#gradient-boosting-trees) and [random forests
](algorithms_and_model_architectures.md#random-forests)





## Entity Recognition


## Question-Answering



# NLP Architectures

## Recent Origins

These papers influenced a paradigm shift towards what will be called [Deep Learning](vocabulary.md#deep-learning
), which will imply the massive adoption of neural networks for ML tasks.

### Word2Vec

"Efficient Estimation of Word Representations in Vector Space" {{ cite mikolov_efficient_2013 }} or simply the
 Word2Vec paper by Mikolov et al. at Google marked a paradigm shift in NLP, as it showed the potential of an embedding
 model trained in large amounts of data (1.6 Billion data words). In particular, they showed the quality of the 
representations obtained after training by using a word similarity task.
A deeper explanation can be found in "word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding
 method" {{ cite goldbert_word2vec_2014 }}
 
[Source code](https://github.com/tmikolov/word2vec)

#

### Elmo

"Deep contextualized word representations" {{ cite peters_deep_2018 }} a.k.a. the "Elmo" paper, improved the results
obtaineed by Word2Vec. The main differencde is that Elmo adds context to word representations. Word vectors are
learned functions of the internal states of a deep bidirectional language model (biLM), pre-trained also on a large
text corpus. This marked the start of the "Sesame Street Saga"

## Attention

This was a game changer paper when it appeared in 2017 {{ cite vaswani_attention_2017 }}.
The concept of attention is taken, as many others, from cognitive sciences (e.g. psycology, neuroscience, education.) It 
describes the process of focusing on certain concrete stimulus/stimuli while ignoring the rest of stimuli in an 
environment. In the case of NLP for example, the context/environment can be a sentence and the stimulus a word.
  
* [Attention and Memory-Augmented Networks for Dual-View Sequential Learning (KDD 2020)]()

#### Sparse Transformers

* [Sparse Tansformer](refs.md#sparse) Self-attention complexity from O(n2) to O(n*sqrt(n)).
* [Reformer](refs.md#reformer) Self-attention complexity O(L2) to O(LlogL), where L is the length of the sequence.
* [Linformer](refs.md#linformer) Self-attention complexity from O(n2) to O(n) in both time and space.


## Transformers

### Sesame Street Saga

The [ELMO](#elmo) paper started a trend to name many NLP model architectures and variations after the characters of
Sesame Street/Muppets. Some refer to this fenomenon as ["Muppetware"](https://www.theverge.com/2019/12/11/20993407/ai-language-models-muppets-sesame-street-muppetware-elmo-bert-ernie)
These are the most relevant ones.  

TODO mention at least ~~ELMo,~~ BERT, Grover, Big BIRD, Rosita, RoBERTa, ERNIEs, and KERMIT. 

#### BERT

By 2018 Google developed what is still as of 2021, the SotA of embedding-based models for the majority of the industry.
Based on the Transformer architecture, it was trained on 3.3 billion words. Comming in two different flavours, base
 and large, they mainly differ on the number of parameters.
 
* [BERT](https://arxiv.org/abs/1810.04805) The reference model for NLP since 2018.
* [SpanBERT](https://arxiv.org/pdf/1907.10529.pdf)
 Masks spans of words instead of random subwords. Spans of words refers to global entities or loca/domain-specific meaning (e.g. American Football)
 Span Boundary Objective(SBO) predicts the span context from boundary token representations. Uses single sentence document-level inputs instead of
 the two sentences in BERT.
 Code: https://github.com/facebookresearch/SpanBERT
* [RoBERTa](https://arxiv.org/abs/1907.11692)
Replication study of BERT pretraining that measures the impact of many key hyperparameters (Bigger Batch size and LR) and training data size (10X).
It shows improvements on most of the SotA results by BERT and followers. Questions the results of some post-BERT models.
It uses a single sentence for the document-level input like SpanBERT.
Code: https://github.com/pytorch/fairseq


#### Small Models/Small Devices
* [Lite transformer with Long-Short Range Attention](refs.md#lite)
Uses Long-Short Range Attention (LSRA) in which a group of heads specializes in
the local context (using convolution) and another group specializes in the
long-distance relationships (ussing the attention mechanism.) Focus on edge (mobile) devices.


<details>
  <summary>Small Models</summary>
 * [Distilbert](https://arxiv.org/abs/1910.01108)
 Check it out in https://huggingface.co/
</details>

<details>
  <summary>Other Sesame Street Papers</summary>

 * [FinBERT](https://arxiv.org/abs/1908.10063) Bert applied to Financial Sentiment Analysis.
 Code: https://github.com/ProsusAI/finBERT]
 * [FinBERT](refs.md#abcd)
</details>

### Non-Sesame Street Environment

** [Turing NLG](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/)
"Turing Natural Language Generation (T-NLG) is a 17 billion parameter language model by Microsoft that outperforms the state of the art on many downstream NLP
tasks. We present a demo of the model, including its freeform generation, question answering, and summarization capabilities, to academics for feedback and
research purposes. <|endoftext|>" - Summary generated by itself.



### Lifelong learning in NLP

* <a name="biesialska">[Continual Lifelong Learning in Natural Language Processing: A Survey](https://www.aclweb.org/anthology/2020.coling-main.574.pdf) Colling, 2020</a>


## Text Generation

Text generation is a subfield of NLP which pursues the automatic generation of human-readable text using
 computational linguistics and AI.
 
Approaches to text generation use a [language model](vocabulary.md#language-model) to generate the probability
 distribution from we can sample to generate the next token in a sentence.
 
One of the most used generative models the so-called is Recurrent Neural Networks (RNN) 