# BERT for sentence pair regression task (Semantic Textual Similarity)
BERT is recently the  state-of-the-art method for NLP tasks.
Here, I use BERT for sentence pair regresssion task.
The reason I do not want to use the BERT sentence classification model is because the sentence pairs scores/ labels are continuous and not integers.
The main task is to use BERT to compute semantic textual similarity between sentence pairs.
The sentences are annotated on a continuos scale [0, 5].
I use BERT with a  regression layer on top.


The English clinical domain data is not publicly available due to privacy reasons.
However, you can use general domain English data from the SemEval STS shared task. This data is saved in the STS_data folder.

Please download the Japanese clinical domain STS data from this <a href="https://github.com/sociocom/Japanese-Clinical-STS" target="_blank">Github repository</a>


For more information about this task check the <a>project website</a> or read the paper shared below.

# Reference
If you use the Japanese dataset please cite our paper:
```
@article{mutinda2021semantic,
  title={Semantic Textual Similarity in Japanese Clinical Domain Texts Using BERT},
  author={Mutinda, Faith Wavinya and Yada, Shuntaro and Wakamiya, Shoko and Aramaki, Eiji},
  journal={Methods of Information in Medicine},
  year={2021},
  publisher={Georg Thieme Verlag KG}
  }
```
