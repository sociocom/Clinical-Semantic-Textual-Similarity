


import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
import tensorflow as tf
import torch, csv
import logging, re
from torch import nn
from pytorch_transformers.modeling_utils import WEIGHTS_NAME, CONFIG_NAME
import os
import pandas as pd
from transformers import BertTokenizer, BertForPreTraining
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
# from transformers import BertJapaneseTokenizer
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertModel, BertPreTrainedModel
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time, datetime, random
import codecs



start_time=time.time()


device_name = tf.test.gpu_device_name()
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are {} GPU(s) available.'.format(torch.cuda.device_count()))
    print('GPU {} will be used'.format(torch.cuda.get_device_name(0)))
else:
    print('There are no GPUs, we will use CPU instead')
    device = torch.device("cpu")


def sequence_truncation(sentence1_token,sentence2_token,max_length):
    while True:
        total_length=len(sentence1_token)+len(sentence2_token)
        if total_length<=max_length:
            break
        if len(sentence1_token)>len(sentence2_token):
            sentence1_token.pop()
        else:
            sentence2_token.pop()


def preprocessingForBert(data:list, tokenizer: BertTokenizer, max_sequence_length=500):
    max_bert_input_length = 0
    for sentence_pair in data:
        sentence_1_tokens, sentence_2_tokens= tokenizer.tokenize(sentence_pair['sentence_1']), tokenizer.tokenize(sentence_pair['sentence_2'])
        sequence_truncation(sentence_1_tokens, sentence_2_tokens, max_sequence_length - 3) #account for positioning tokens

        max_bert_input_length = max(max_bert_input_length, len(sentence_1_tokens) + len(sentence_2_tokens) + 3)
        sentence_pair['sentence_1_tokens'] = sentence_1_tokens
        sentence_pair['sentence_2_tokens'] = sentence_2_tokens

    data_input_ids = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
    data_token_type_ids = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
    data_attention_masks = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
    data_scores = torch.empty((len(data), 1), dtype=torch.float)

    for index, sentence_pair in enumerate(data):
        tokens = []
        input_type_ids = []

        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in sentence_pair['sentence_1_tokens']:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        for token in sentence_pair['sentence_2_tokens']:
            tokens.append(token)
            input_type_ids.append(1)
        tokens.append("[SEP]")
        input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_masks = [1] * len(input_ids)
        while len(input_ids) < max_bert_input_length:
            input_ids.append(0)
            attention_masks.append(0)
            input_type_ids.append(0)

        data_input_ids[index] = torch.tensor(input_ids, dtype=torch.long)
        data_token_type_ids[index] = torch.tensor(input_type_ids, dtype=torch.long)
        data_attention_masks[index] = torch.tensor(attention_masks, dtype=torch.long)

        if 'score' not in sentence_pair or sentence_pair['score'] is None:
            data_scores[index] = torch.tensor(float('nan'), dtype=torch.float)  #dtype=torch.float torch.long
        else:
            data_scores[index] = torch.tensor(sentence_pair['score'], dtype=torch.long)  #dtype=torch.float

    return data_input_ids, data_token_type_ids, data_attention_masks, data_scores


def loadData(file):
    read_file=codecs.open(file,'r',encoding='utf-8')
    data = []
    for index,lines in enumerate(read_file):
        line=tuple(lines.split('\t'))
        data.append({'index':index, 'sentence_1':line[0],'sentence_2':line[1],'score':int(float(line[2]))})
    return data


def pearson_accuracy(predictions,true_scores,scores_len):
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    flat_true_scores = np.asarray(true_scores)
    flat_true_scores = flat_true_scores.reshape(1,scores_len)
    flat_true_labels = [int(item) for sublist in flat_true_scores for item in sublist]
    flat_true_labels = np.asarray(flat_true_labels)

    pearsonScore = pearsonr(flat_true_labels, flat_predictions)
    accuracy=accuracy_score(flat_true_labels,flat_predictions)

    print('pearson correlation score is: {}'.format(pearsonScore))
    print('accuracy is: {}'.format(accuracy))
    return pearsonScore,accuracy, flat_predictions, flat_true_labels



class ClassificationModel():
    def __init__(self,train_file,dev_file,bertModel,epochs,train_batchsize,test_batchsize,outputdir,save_csv,epoch_name,
                 learning_rate,warm_up):

        self.bertModel=bertModel
        self.train_file=train_file
        self.dev_file=dev_file
        self.train_batchsize, self.test_batchsize=train_batchsize,test_batchsize
        self.epochs = epochs
        self.learning_rate = learning_rate#2e-5
        self.eps = 1e-8
        self.warmup_steps = warm_up#10
        self.seed_value = 100
        self.outputdir = outputdir
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bertModel)
        self.save_csv=save_csv
        self.epoch_name=epoch_name


    def train(self):
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)

        train=loadData(self.train_file)
        train_input_id, train_token_type_id, train_attmask, train_scores = preprocessingForBert(train, self.bert_tokenizer)
        train_data = TensorDataset(train_input_id, train_token_type_id, train_attmask, train_scores)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.train_batchsize)


        self.model = BertForSequenceClassification.from_pretrained(self.bertModel, num_labels=6,
                                                              output_attentions=False,output_hidden_states=False)
        self.model.cuda()

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-8)
        total_steps = len(train_dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps,num_training_steps=total_steps)

        loss_values = []
        t0 = time.time()

        avg_pearson,avg_accuracy=0,0
        for epoch_i in range(0, self.epochs):
            print('======== Epoch {} / {} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            total_loss = 0
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                if step % 200 == 0 and not step == 0:
                    elapsed = time.time() - start_time
                    print('  Batch {}  of  {}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                b_input_ids = batch[0].to(device)
                b_token_type_id = batch[1].to(device)
                b_input_mask = batch[2].to(device)
                b_labels = batch[3].to(device, dtype=torch.int64)

                self.model.zero_grad()
                outputs =self. model(b_input_ids, token_type_ids=b_token_type_id,attention_mask=b_input_mask, labels=b_labels)

                loss = outputs[0]
                total_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)
            loss_values.append(avg_train_loss)

            print(" Average training loss: {0:.2f}".format(avg_train_loss))
            print(" Epoch time: {}".format(time.time() - t0))

            print("Running Validation --------")

            pearson_score,accuracy_scor=self.Model_eval(epoch_i=str(epoch_i))

            with open(self.save_csv, mode="a" ) as f:
                writer = csv.writer(f)
                writer.writerow([str(epoch_i)+'_'+self.epoch_name,pearson_score[0],accuracy_scor])



        print("Training complete!")
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.outputdir)
        self.bert_tokenizer.save_pretrained(self.outputdir)

        return self.model

    def Model_eval(self,epoch_i):
        dev= loadData(self.dev_file)

        dev_input_id, dev_token_type_id, dev_attmask, dev_scores = preprocessingForBert(dev,self.bert_tokenizer)

        validation_data = TensorDataset(dev_input_id, dev_token_type_id, dev_attmask, dev_scores)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.test_batchsize)

        print('\n len validation file: {}'.format(len(dev_input_id)))

        self.model.eval()

        t0 = time.time()
        predictions, true_scores = [], []
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_token_type_id, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=b_token_type_id,
                                attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions.append(logits)
            true_scores.append(label_ids)

        pearsonScore,accuracy, flat_predictions, flat_true_labels=pearson_accuracy(predictions,true_scores,len(dev_input_id))

        results_df = pd.DataFrame()
        results_df["sentence_1"] = np.array([instance["sentence_1"] for instance in dev])
        results_df["sentence_2"] = np.array([instance["sentence_2"] for instance in dev])

        results_df['true_scores'] = np.array(flat_true_labels)
        results_df['predictions'] = np.array(flat_predictions)

        results_df.to_csv(self.outputdir+'/{}_{}_classification_results.csv'.format(epoch_i,str(time.time())), index=False,encoding='utf_8_sig')

        print("Validation time was : {}".format(time.time() - t0))

        return pearsonScore,accuracy



def sentence_predict(sentences,model,batch_size):
    bert_tokenizer = BertTokenizer.from_pretrained(bertModel)

    input_ids, token_type_ids_eval, attention_masks_eval, correct_scores_eval = preprocessingForBert([
        {'sentence_1': s1, 'sentence_2': s2} for s1, s2 in sentences], bert_tokenizer)

    prediction_data = TensorDataset(input_ids, token_type_ids_eval, attention_masks_eval, correct_scores_eval)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    model.eval()
    prediction = []
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=token_type_ids_eval,attention_mask=attention_masks_eval)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()

    prediction.append(logits)
    flat_prediction = [item for sublist in prediction for item in sublist]
    flat_prediction = np.argmax(flat_prediction, axis=1).flatten()

    return flat_prediction





epochs=10
te_batch=3
bertModel='bert-base-uncased'

batch=2
lr=2e-5
warm=5

train_file='STS_data/train.txt'
test_file = 'STS_data/test.txt'

output_dir='output/sts_classification'
save_file='output/sts_classification.csv'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


model=ClassificationModel(train_file=train_file,dev_file=test_file,bertModel=bertModel,epochs=epochs,
                          train_batchsize=batch,test_batchsize=te_batch,outputdir=output_dir,
                          save_csv=save_file,epoch_name=((str(batch)+'_'+str(lr)+'_'+str(warm))),learning_rate=lr,
                          warm_up=warm)
model.train()



print('\n\n total time', time.time()-start_time)



