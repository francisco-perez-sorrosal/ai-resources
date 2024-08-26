# Applications

* [DeepFace](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) always remind me of how Schwarzenegger is framed for the murder of almost a hundred unarmed civilians in [The Running Man](https://www.imdb.com/title/tt0093894/)

## Prediction

## Regression

## Classification

## Personalization/Recommender systems

Make decisions is a key part of what we consider intelligence. Learn from limited samples to make good decissions.

Multi-arm banditds (Contextual) Subcase of RL

Emma Brunskill

[Personalization, Recommendation and Search (PRS) Workshop by Netflix](https://prs2021.splashthat.com/)

## Recognition (images, audio/speech, text)

Initial techniques for image/audio/text recognition were based on lots of feature engineering that were fed to a
classical ML model such as SVM. With the advent of DL these features have been discovered automatically by using
huge datasets such as ImageNet and NN architectures based on convolutions. And further that, we now are in the
position of searching automatically for more efficient architectures (meta-architecture discovery).

In the same way, speech recognition used to require a lot of experts in language, preprocessing, Gaussian or Hidden
Markov models etc. but with the democratization of NN almost all those techniques are not needed anymore.
 


## Computer vision

## Clustering and anomaly detection

## Natural language processing, generation, and understanding

For a more in depth analysis of the SotA in NLP see [the corresponding chapter](nlp.md).

## Translation

Being one of the initial fields for research in NLP, traditionally, machine translation was based on sentence-based
 statistical techniques. With the advent of big data and computing power, neural networks have taken over the field. 

For example, along 2016/2017 Google switched from sentence-based/linguistic expert-based algorithmic approach to deep
-learning based methods (what is called Neural Machine Translation, or NMT.) The leap in quality of the transations
 was massive:

- [Found in translation: More accurate, fluent sentences in Google Translate (Barak Turovsky, Google Translate Product Lead, Nov 2016](https://blog.google/products/translate/found-translation-more-accurate-fluent-sentences-google-translate/)
- [Higher quality neural translations for a bunch more languages (Barak Turovsky, Google Translate Product Lead, Mar 2017)](https://www.blog.google/products/translate/higher-quality-neural-translations-bunch-more-languages/)
- [A Neural Network for Machine Translation, at Production Scale](https://ai.googleblog.com/2016/09/a-neural-network-for-machine.html)
- [The Shallowness of Google Translate (Douglas Hofstadter)](https://www.theatlantic.com/technology/archive/2018/01/the-shallowness-of-google-translate/551570/)

Most of those models are also multilingual, meaning that a single model is capable of translating from any source
 language to any target language.

## Financial

Models like FinBERT have been finetuned with a large financial corpora of articles {{ #cite yang_finbert_2020 }} 

## Medicine

Dermatology - Detect skin cancer or problematic skin-related diseases.

## Legal/Law


## Gaming

Games have been always been a center of attention for AI, being backgamon, chess or Go, probably the most popular
 examples. Initially, most of the approaches were rule-based expert systems. There were some exceptions such as
  {{ #cite sejknowsy_deep_2018 }} Since the success of Deep Blue and more recently AlphaGo and AlphaZero beating grandmasters, the
  field of AI/ML has even had more impact.

But the success in board games is not only the only focus of attention of the game industry. StarCraft is another
popular game strategy/intereactive game where the researchers in AI have done amazing improvements in playing against
humans.

### Programing Languages

* [Automated Coding article (O'Reilly)](https://www.oreilly.com/radar/automated-coding-and-the-future-of-programming/?sfmc_id=85378584&utm_medium=email&utm_source=platform+b2b&utm_campaign=engagement&utm_content=whats+new+thinking+20200831)
* [Unsupervised Translation of Programming Languages](https://arxiv.org/pdf/2006.03511.pdf)


### Databases

Even traditional fields of Computer Science such as DBMSs can't escape from the influence of AI these days. In
 {{ #cite krasa_case_2017 }} the authors
replace DBMS core components with NNs being able to improve the performance of caches of classical data structures
 for data management -such as B-Trees- while using less system resources such as memory/disk space.
