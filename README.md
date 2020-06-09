# StorylineNLP
A Curriculum-Style Introduction on Computational Linguistics

### Outline:
[1. Word Representations](https://github.com/warnikchow/storylineNLP#word-representations)</br>
[2. Tokenization and Character-Level Modeling](https://github.com/warnikchow/storylineNLP#tokenization-and-character-level-modeling)</br>
[3. Classical NLP Pipeline](https://github.com/warnikchow/storylineNLP#classical-nlp-pipeline)</br>
[4. Corpus Construction](https://github.com/warnikchow/storylineNLP#corpus-construction)</br>
[5. Sentence-Level Analysis](https://github.com/warnikchow/storylineNLP#sentence-level-analysis)</br>
[6. Document-Level Analysis](https://github.com/warnikchow/storylineNLP#document-level-analysis)</br>
[7. Attention Models for Translation and Generation](https://github.com/warnikchow/storylineNLP#attention-models-for-translation-and-generation)</br>
[8. Unsupervised Pretrained LMs and Transfer Learning](https://github.com/warnikchow/storylineNLP#unsupervised-pretrained-lms-and-transfer-learning)</br>
[9. Dialog and Conversation](https://github.com/warnikchow/storylineNLP#dialog-and-conversation)

### References:

- Repositories on the awesome NLP papers</br>
https://github.com/mhagiwara/100-nlp-papers</br>
https://github.com/THUNLP-MT/MT-Reading-List</br>
- Nice link for neural conversational models</br>
https://github.com/csnlp/Dialogue-Generation</br>
- Well-described NLP blog (in Korean)</br>
https://wikidocs.net/21667

### Acronyms in the Body:

- JMLR: Journal of Machine Learning Research
- TACL: Transactions of the Association for Computational Linguistics 
- CL: Computational Linguistics
- NIPS: Neural Information Processing Systems (currently NeurIPS)
- ICML: International Conference on Machine Learning 
- ICLR: International Conference on Learning Representations
- AAAI: Conference on Artificial Intelligence, by Association for the Advancement of Artificial Intelligence
- ICASSP: International Conference on Acoustics, Speech, and Signal Processing
- ACL: Annual Meeting of the Association for Computational Linguistics 
- NAACL: North American Chapter of the Association for Computational Linguistics
- EMNLP: Conference on Empirical Methods in Natural Language Processing
- SNL: The Workshop on Speech and Natural Language

#### Featured NLP-friendly Venues

- **EMNLP, ACL, NAACL, EACL, AACL, IJCNLP, CoLing, CoNLL** (NLP to CL, in order; quite subjective)
- **ICASSP, Interspeech, ASRU, SLT** (Speech and spoken language processing)
- **LREC, SemEval, WMT, IWSLT** (Resources and evaluation)
- **SIGIR, KDD, ICDM, CIKM** (Information retrieval)
- **WSDM, ICWSM** (Web and social media)

## Word Representations

- Bengio et al., [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/v3/bengio03a.html?source=post_page---------------------------), JMLR, 2003. **[RNNLM]**
- Mikolov et al., [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and), NIPS 2013. **[word2vec]**
- Mikolov et al., [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781), ICLR 2013. **[Analogy Test]**
- Pennington et al., [GloVe: Global Vectors for Word Representation](https://www.aclweb.org/anthology/D14-1162/), EMNLP 2014. **[GloVe]**
- Bojanowski et al., [Enriching Word Vectors with Subword Information](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00051), TACL, 2017. **[fastText]**

## Tokenization and Character-Level Modeling

- Schuster and Nakajima, [Japanese and korean voice search](https://ieeexplore.ieee.org/abstract/document/6289079), ICASSP 2012. **[WPM]**
- Sennrich et al., [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), ACL 2016. **[BPE]**
- Kudo, [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959), ACL 2018. **[Subword]**
- Kudo and Richardson, [SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226), EMNLP 2018. **[SentencePiece]**
- Kim et al., [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615), AAAI 2016. **[Character-level]**
- Zhang and Lecun, [Which Encoding is the Best for Text Classification in Chinese, English, Japanese and Korean?](https://arxiv.org/abs/1708.02657)

## Classical NLP Pipeline 

- Collobert et al., [Natural Language Processing (almost) from Scratch](https://arxiv.org/abs/1103.0398), JMLR, 2011.

#### xxBanks and Methods
- Marcus et al., [Building a large annotated corpus of English: The Penn Treebank](https://repository.upenn.edu/cis_reports/237/), 1993. **[PTB]**
- Palmer et al., [The proposition bank: An annotated corpus of semantic roles](https://www.mitpressjournals.org/doi/abs/10.1162/0891201053630264), CL, 2005. **[PropBank]**
- Strubell et al., [Linguistically-Informed Self-Attention for Semantic Role Labeling](https://arxiv.org/abs/1804.08199), EMNLP 2018. **[LISA]**
- Tenney et al., [BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/abs/1905.05950), ACL 2019.

## Corpus Construction

- Pustejovsky and Stubbs, [Natural Language Annotation for Machine Learning](https://doc.lagout.org/science/Artificial%20Intelligence/Machine%20learning/Natural%20Language%20Annotation%20for%20Machine%20Learning_%20A%20Guide%20to%20Corpus-...%20%5BPustejovsky%20%26%20Stubbs%202012-11-04%5D.pdf), O'Reilly, 2012.

#### Sentiment
- Bo and Lee, [A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts](https://dl.acm.org/doi/10.3115/1218955.1218990), ACL 2004. **[Subjectivity]**
- Maas et al., [Learning Word Vectors for Sentiment Analysis](https://dl.acm.org/doi/10.5555/2002472.2002491), ACL 2011. **[IMDB]**
- Socher et al., [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://www.aclweb.org/anthology/D13-1170/), EMNLP 2013. **[SST]**

#### Intent
- Hemphill et al., [The ATIS Spoken Language Systems Pilot Corpus](https://www.aclweb.org/anthology/H90-1021/), SNL 1990. **[ATIS]**
- Stolcke et al., [Dialogue act modeling for automatic tagging and recognition of conversational speech](https://www.mitpressjournals.org/doi/abs/10.1162/089120100561737), CL, 2000. **[SWBD Dialog Act]**
- Lugosch et al., [Speech Model Pre-training for End-to-End Spoken Language Understanding](https://arxiv.org/abs/1904.03670), Interspeech 2019. **[Fluent Speech Command]**

## Sentence-Level Analysis

- Kim, [Convolutional neural networks for sentence classification](https://arxiv.org/abs/1408.5882), EMNLP 2014. **[KimCNN]**
- Zhang et al., [Character-level Convolutional Networks for Text Classification](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classifica), NIPS 2015. **[CharCNN]**
- Tang et al., [Document modeling with gated recurrent neural network for sentiment classification](https://www.aclweb.org/anthology/D15-1167/), EMNLP 2015. 
- Liu and Lane, [Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling](https://arxiv.org/abs/1609.01454), Interspeech 2016. 
- Lin et al., [A Structured Self-Attentive Sentence Embedding](https://arxiv.org/abs/1703.03130), ICLR 2017. 

## Document-Level Analysis

#### MRQA
- Yang et al., [WikiQA: A Challenge Dataset for Open-Domain Question Answering](https://www.aclweb.org/anthology/D15-1237/), ACL 2015. **[WikiQA]**
- Rajpurkar et al., [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250), EMNLP 2016. **[SQuAD]**
- Chen et al., [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051), ACL 2017.
- Xiong et al., [Dynamic Coattention Networks For Question Answering](https://arxiv.org/abs/1611.01604), ICLR 2017.

#### Summarization
- Mihalcea and Tarau, [TextRank: Bringing Order into Texts](https://www.aclweb.org/anthology/W04-3252/), EMNLP 2004. **[TextRank]**
- Rush et al., [A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/abs/1509.00685), EMNLP 2015.
- See et al., [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368), ACL 2017. **[Pointer-Generator]**

## Attention Models for Translation and Generation

- Sutskever et al., [Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-), NIPS 2014. **[Seq2Seq]**
- Cho et al., [Learning phrase representations using RNN encoder-decoder for statistical machine translation](https://arxiv.org/abs/1406.1078), EMNLP 2014. **[RNN Encoder-decoder]**
- Bahdanau et al., [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), ICLR 2015. **[Bahdanau Attention]**
- Luong et al., [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025), EMNLP 2015. **[Luong Attention]**
- Wu et al., [Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144), 2016. **[GNMT]**
- Johnson et al., [Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00065), TACL, 2017. **[Google Multilingual NMT]**
- Artetxe et al., [Unsupervised Neural Machine Translation](https://arxiv.org/abs/1710.11041), ICLR 2018. **[Unsupervised NMT]**
- Vaswani et al., [Attention Is All You Need](http://papers.nips.cc/paper/7181-attention-is-all-you-need), NIPS 2017. **[Transformer, Self-attention]**

## Unsupervised Pretrained LMs and Transfer Learning

- Wang et al., [Glue: A multi-task benchmark and analysis platform for natural language understanding](https://arxiv.org/abs/1804.07461), ICLR 2019. **[GLUE]**
- Peters et al., [Deep contextualized word representations](https://arxiv.org/abs/1802.05365), NAACL 2018. **[ELMo]**
- Radford et al., [Improving Language Understanding by Generative Pre-Training](https://openai.com/blog/language-unsupervised/), 2018. **[OpenAI GPT]**
- Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), NAACL 2019. **[BERT]**
- Song et al., [MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/abs/1905.02450), ICML 2019. **[MASS]**
- Radford et al., [Language Models are Unsupervised Multitask Learners](https://openai.com/blog/better-language-models/), 2019. **[OpenAI GPT2]**

## Dialog and Conversation

- Vinyals and Le, [A Neural Conversational Model](https://arxiv.org/abs/1506.05869), ICML Deep Learning Workshop, 2015. **[NCM]**
- Adiwardana et al., [Towards a Human-like Open-Domain Chatbot](https://arxiv.org/abs/2001.09977), 2020. **[Meena]**
