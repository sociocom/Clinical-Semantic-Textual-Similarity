#this code is copied from: https://www.kaggle.com/tenffe/transform-tf-bert-to-pytorch-bert


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertJapaneseTokenizer
from transformers import BertForSequenceClassification
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
import torch, os, shutil


path='UTH_BERT_BASE_MC_BPE_V25000_10M/'


def transform_bert_to_dir(BERT_MODEL_PATH, WORK_DIR, bert_model_name):
    if not os.path.exists(WORK_DIR):
        os.mkdir(WORK_DIR)
    convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(BERT_MODEL_PATH + 'model.ckpt-10000000',
                                                                  BERT_MODEL_PATH + 'bert_config.json',
                                                                  WORK_DIR + f'pytorch_model_{bert_model_name}.bin')
    shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', WORK_DIR + f'bert_config_{bert_model_name}.json')


transform_bert_to_dir(BERT_MODEL_PATH=path,WORK_DIR=path+'dump/',bert_model_name='uth')


bert_model=path+'dump/'
model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=4,
                                                      output_attentions=False,output_hidden_states=False)
print(model)
