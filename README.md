# BERT for sentence pair regression task (Semantic Textual Similarity)
BERT is recently the  state-of-the-art method for NLP tasks.
Here, I use BERT for sentence pair regresssion task.
The reason I do not want to use the BERT sentence classification model is because the sentence pairs scores/ labels are continuous and not integers.
The main task is to use BERT to compute semantic textual similarity between sentence pairs.
The sentences are annotated on a continuos scale [0, 5].
I use BERT with a  regression layer on top.
