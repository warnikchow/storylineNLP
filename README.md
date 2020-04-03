# StorylineNLP
A Curriculum-Style Introduction on Computational Linguistics

### Outline:
[1. Word Representations](https://github.com/warnikchow/storylineNLP#word-representations)</br>
[2. Tokenization and Character-Level Modeling](https://github.com/warnikchow/storylineNLP#tokenization-and-character-level-modeling)</br>
[3. Classical NLP Pipeline](https://github.com/warnikchow/storylineNLP#classical-nlp-pipeline)</br>
[4. Sentence-Level Analysis](https://github.com/warnikchow/storylineNLP#sentence-level-analysis)</br>
[5. Document-Level Analysis](https://github.com/warnikchow/storylineNLP#document-level-analysis)</br>
[6. Attention Models for Translation and Generation](https://github.com/warnikchow/storylineNLP#attention-models-for-translation-and-generation)</br>
[7. Unsupervised Pretrained LMs and Transfer Learning](https://github.com/warnikchow/storylineNLP#unsupervised-pretrained-lms-and-transfer-learning)</br>

### References:

- Repositories on the awesome NLP papers</br>
https://github.com/mhagiwara/100-nlp-papers</br>
https://github.com/THUNLP-MT/MT-Reading-List</br>
- Well-described NLP blog (in Korean)</br>
https://wikidocs.net/21667

### Acronyms:

- JMLR: Journal of Machine Learning Research
- TACL: Transactions of the Association for Computational Linguistics 
- CL: Computational Linguistics
- NIPS: Neural Information Processing Systems (currently NeurIPS)
- ICLR: International Conference on Learning Representations
- ICASSP: International Conference on Acoustics, Speech, and Signal Processing
- ACL: Annual Meeting of the Association for Computational Linguistics 
- NAACL: North American Chapter of the Association for Computational Linguistics
- EMNLP: Conference on Empirical Methods in Natural Language Processing
- SNL: The Workshop on Speech and Natural Language

## Word Representations

- Bengio et al., A Neural Probabilistic Language Model, JMLR, 2003. [RNNLM]
- Mikolov et al., Distributed Representations of Words and Phrases and their Compositionality, NIPS 2013. [word2vec]
- Mikolov et al., Efficient Estimation of Word Representations in Vector Space, ICLR 2013.
- Pennington et al., GloVe: Global Vectors for Word Representation, EMNLP 2014. [GloVe]
- Bojanowski et al., Enriching Word Vectors with Subword Information, TACL, 2017. [fastText]

## Tokenization and Character-Level Modeling

- Sennrich et al., Neural Machine Translation of Rare Words with Subword Units. ACL 2016. [BPE]
- Schuster and Nakajima, Japanese and korean voice search. ICASSP 2012. [WPM]
- Kudo, Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates, 2018.
- Zhang et al., Character-level Convolutional Networks for Text Classification, NIPS 2015. [charCNN]

## Classical NLP Pipeline 

- Collobert et al., Natural Language Processing (almost) from Scratch, JMLR, 2011.
- Marcus et al., Building a large annotated corpus of English: The Penn Treebank, 1993. [PTB]
- Palmer et al., The proposition bank: An annotated corpus of semantic roles, CL, 2005. [PropBank]
- Strubell et al., Linguistically-Informed Self-Attention for Semantic Role Labeling, EMNLP 2018. [LISA]

## Sentence-Level Analysis

- Pustejovsky and Stubbs, Natural Language Annotation for Machine Learning, O'Reilly, 2012.
- Maas et al., Learning Word Vectors for Sentiment Analysis, ACL 2011. [IMDB]
- Socher et al., Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank, EMNLP 2013. [SST]
- Stolcke et al., Dialogue act modeling for automatic tagging and recognition of conversational speech, CL, 2000. [SWBD DA]
- Hemphill et al., The ATIS spoken language systems pilot corpus, SNL 1990. [ATIS]
- Kim, Convolutional neural networks for sentence classification, EMNLP 2014. [KimCNN]
- Liu and Lane, Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling, Interspeech 2016.

## Document-Level Analysis

- Tang et al., Document modeling with gated recurrent neural network for sentiment classification, EMNLP 2015. 
- Rajpurkar et al., SQuAD: 100,000+ Questions for Machine Comprehension of Text, EMNLP 2016. [SQuAD]
- Chen et al., Reading Wikipedia to Answer Open-Domain Questions, ACL 2017.
- Xiong et al., Dynamic Coattention Networks For Question Answering, ICLR 2017.
- Mihalcea and Tarau, TextRank: Bringing Order into Texts, EMNLP 2004.
- Rush et al., A Neural Attention Model for Abstractive Sentence Summarization, EMNLP 2015.
- See et al., Get To The Point: Summarization with Pointer-Generator Networks, ACL 2017. [Pointer-Generator]

## Attention Models for Translation and Generation

- Sutskever et al., Sequence to Sequence Learning with Neural Networks, NIPS 2014. [Seq2Seq]
- Cho et al., Learning phrase representations using RNN encoder-decoder for statistical machine translation, EMNLP 2014. [RNN Encoder-decoder]
- Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate, ICLR 2015. [Bahdanau Attention]
- Luong et al., Effective Approaches to Attention-based Neural Machine Translation, EMNLP 2015. [Luong Attention]
- Wu et al., Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation, 2016. [GNMT]
- Johnson et al., Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation, TACL, 2017. [Google Multilingual NMT]
- Artetxe et al., Unsupervised Neural Machine Translation, ICLR 2018. 
- Vaswani et al., Attention Is All You Need, NIPS 2017. [Transformer]

## Unsupervised Pretrained LMs and Transfer Learning

- Wang et al., Glue: A multi-task benchmark and analysis platform for natural language understanding, ICLR 2019. [GLUE]
- Peters et al., Deep contextualized word representations, NAACL 2018. [ELMo]
- Radford et al., Improving Language Understanding by Generative Pre-Training, 2018. [OpenAI GPT]
- Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL 2019. [BERT]
- Tenney et al., BERT Rediscovers the Classical NLP Pipeline, ACL 2019.
- Song et al., MASS: Masked Sequence to Sequence Pre-training for Language Generation, ICML 2019. [MASS]
- Radford et al., Language Models are Unsupervised Multitask Learners, 2019. [OpenAI GPT2]
- Adiwardana et al., Towards a Human-like Open-Domain Chatbot, 2020. [Meena]
