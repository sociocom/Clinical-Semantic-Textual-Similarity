# Clinical Semantic Textual Similarity (English& Japanese)
The scripts are for calculating semantic similarity in clinical/biomedical domain texts.
The model input is sentence pairs annotated with semantic similarity scores between 0 (low semantic similarity) and 5 (high semantic similarity)

Depending on the dataset, the sentence pairs are annotated with discrete semantic similarity scores [0,1,2,3,4,5] or continuous scores [0-5].
For the discrete scores, we use BERT model for sequence classification.
For the continuous scores, we use the standard BERT model and add a regression layer on top.



# Dataset
The English clinical domain data is not publicly available due to privacy reasons.
However, you can use general domain English data from the SemEval STS shared task. This data is saved in the STS_data folder.

Japanese clinical domain STS data can be downloaded freely from this <a href="https://github.com/sociocom/Japanese-Clinical-STS" target="_blank">Github repository</a>


# BERT Models

### Japanese models
General domain BERT: <a href="https://github.com/cl-tohoku/bert-japanese">https://github.com/cl-tohoku/bert-japanese</a>

Clinical domain BERT: <a href="https://ai-health.m.u-tokyo.ac.jp/uth-bert"> https://ai-health.m.u-tokyo.ac.jp/uth-bert</a>

### English models
Clinical BERT: <a href="https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT">https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT</a>

SciBERT: <a href="https://github.com/allenai/scibert">https://github.com/allenai/scibert</a>

BioBERT: <a href="https://github.com/dmis-lab/biobert">https://github.com/dmis-lab/biobert</a>






# Reference
For more information about this task check the <a href="" target="_blank">project website</a> and read the paper below.

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
