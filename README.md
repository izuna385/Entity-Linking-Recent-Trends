# Recent trend(EMNLP'19, CoNLL'19, and others.)

* Fine-Grained Evaluation for Entity Linking (EMNLP'19)

  * [repo](https://github.com/henryrosalesmendez/EL_exp)

[]()

* Learning Dynamic Context Augmentation for Global Entity Linking (EMNLP'19)

  * [paper](https://arxiv.org/abs/1909.02117), [repo](https://github.com/YoungXiyuan/DCA)

[]()

* EntEval: A Holistic Evaluation Benchmark for Entity Representations (EMNLP '19)

  * [repo](https://github.com/ZeweiChu/EntEval)


[]()

* Fine-Grained Entity Typing for Domain Independent Entity Linking

  * [paper](https://arxiv.org/abs/1909.05780)

* Investigating Entity Knowledge in BERT With Simple Neural End-To-End Entity Linking	(CoNLL '19)

* Learning Dense Representations for Entity Retrieval (CoNLL '19')

  * [paper](https://arxiv.org/abs/1909.10506), [repo](https://github.com/google-research/google-research/tree/master/dense_representations_for_entity_retrieval/)

  * They proposed no use of alias table(which was based on wikipedia statistics or prepared one) and searching all entities by brute-force/approximate nearest search for linking entity per mention.

# Recent trend(~ACL'19)
* Trends of  leveraging all information(e.g. mention's type and definition and documents in which mention exists, etc...) seems to be disappering.

* Although Wikipedia domain can use its hyperlink(=mention-entity pairs, about 7,500,000) for training linking model, under some domain-specific situations there are not so much mention-entity pairs.

* Therefore, some papers are now challenging distant-learning and zero-shot learning of Entity linking.

  * Distant Learning

    * [Distant Learning for Entity Linking with Automatic Noise Detection](https://github.com/izuna385/papers/wiki/038_Distant_Learning_for_Entity_linking(ACL19))

      * [slides](https://speakerdeck.com/izuna385/distant-learning-for-entity-linking-with-automatic-noise-detection)

      * They proposed framing EL as Distant Learning problem, in which no labeled training data is available, and de-noising model for this task.

    * [Boosting Entity Linking Performance by Leveraging Unlabeled Documents](https://arxiv.org/abs/1906.01250)

  * Zero-shot Linking

    * [Zero-Shot Entity Linking by Reading Entity Descriptions](https://arxiv.org/abs/1906.07348)

    * [slides](https://speakerdeck.com/izuna385/zero-shot-entity-linking-by-reading-entity-descriptions)

    * They proposed Zero-shot EL, under which no test mentions can be seen during training. For tackling Zero-shot EL, they proposed Domain-adaptive strategy for pre-training Language model. Also, they showed that mention-entity description cross-attention is crucial for EL.

# Recent Baselines(~ACL'18)
* *Bold style* indicates its SoTA score of a specific dataset.

| Baseline models                                                                                      | Year      | Dataset                                                              | code                          | Run?                          | Code address                                                                                                                                                                |
|------------------------------------------------------------------------------------------------------|-----------|----------------------------------------------------------------------|-------------------------------|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Entity Linking via Joint Encoding of Types,Descriptions,and Context                                  | EMNLP2017 | CoNLL-YAGO(82.9,acc),ACE2004,ACE2005,WIKI(**89.0**,f1)               | Tensorflow                    | Only Traind model is uploaded | [here](https://nitishgupta.github.io/neural-el/)                                                                                                                            |
| ┗ (Very Similar to the above) Joint Multilingual Supervision for Cross-lingual Entity Linking        | EMNLP2018 | TH-Test,McN-Test,TAC2015                                             | Pytorch                       | Checking                      | [here](https://github.com/shyamupa/xling-el)                                                                                                                                |
| Neural Collective Entity Linking(NCEL)                                                               | CL2018    | CoNLL-YAGO, ACE2004, AQUAINT,TAC2010(**91.0**,mic-p),WW              | pytorch                       | Bug                           | [here](https://github.com/TaoMiner/NCEL)                                                                                                                                    |
| Improving Entity Linking by Modeling Latent Relations between Mentions                               | ACL2018   | CoNLL-YAGO(**93.07**,mic-acc),AQUAINT,ACE2004,CWEB,WIKI(84.05,f1)    | pytorch                       | Evaluation Done               | [here](https://github.com/lephong/mulrel-nel)                                                                                                                               |
| ELDEN                                                                                                | NAACL2018 | CoNLL-PPD(93.0,p-mic),TAC2010(89.6,mic-p)                            | lua,torch(lua)                | Bug                           | [here](https://github.com/priyaradhakrishnan0/ELDEN)                                                                                                                        |
| Deep Joint Entity Disambiguation with Local Neural Attention                                         | EMNLP2017 | CoNLL-YAGO(92.22,mic-acc),CWEB,WW,ACE2004,AQUAINT,MSNBC              | lua,torch(lua)                | Train Running(2019/01/15)     | [here](https://github.com/dalab/deep-ed)                                                                                                                                    |
| Hierarchical Losses and New Resources for Fine-grainid Entity Typing and Linking                     | ACL2018   | Medmentions,Typenet                                                  | pytorch                       | Bug                           | [here](https://github.com/MurtyShikhar/Hierarchical-Typing)                                                                                                                 |
| Joint Learning of the Embedding of Words and Entities for Named Entity Disambiguation(Yamada,Shindo) | CoNLL2016 | CoNLL-YAGO(91.5,mic-acc),CoNLL-PPD(93.1,p-mic),TAC2010(85.5,mic-acc) | pytorch/Tensorflow(original), | checking                      | [Baseline(2016)](https://github.com/hiroshi-ho/EDPipline),[Baseline Original](https://github.com/wikipedia2vec/wikipedia2vec)                                               |
| Learning Distributed Representations of Texts and Entities from Knowledge Base(Yamada,Shindo)        | ACL2017   | CoNLL-PPD(**94.7**,p-mic),TAC2010(87.7,mic-acc)                      | pytorch/Keras(original)       | checking                      | [Torch](https://github.com/lephong/mulrel-nel/blob/master/nel/ntee.py), [Torch](https://github.com/AdityaAS/PyTorch_NTEE), [Original](https://github.com/studio-ousia/ntee) |

## Entity Linking Introductions
<img src='./img/intro.png' width=960>

<img src='./img/procedure.png' width=960>

## Local model and Global model
* Details are wrintten in *Neural Collective Entity Linking*. [paper](http://www.aclweb.org/anthology/C18-1057)

### Trend in the point of *local* vs *global*

<img src='./img/localvsglobal.png' width=960>

### What is *local*/*global* model?
<img src='./img/local.png' width=960>

<img src='./img/global.png' width=960>
