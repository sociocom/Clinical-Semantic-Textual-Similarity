
#some parts of this code are copied from several gihub projects
#truncation and bert preprocessing is copied from https://github.com/huggingface/pytorch-pretrained-BERT/blob/78462aad6113d50063d8251e27dbaadb7f44fbf0/examples/extract_features.py#L150
#and other parts are from https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#tfbertforpretraining, https://mccormickml.com/2019/07/22/BERT-fine-tuning/

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from scipy.stats import pearsonr
import tensorflow as tf
import torch
import logging
from torch import nn
from pytorch_transformers.modeling_utils import WEIGHTS_NAME, CONFIG_NAME
import os
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
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


def preprocessingForBert(data:list, tokenizer: BertTokenizer, max_sequence_length=220):
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
            data_scores[index] = torch.tensor(float('nan'), dtype=torch.float)
        else:
            data_scores[index] = torch.tensor(sentence_pair['score'], dtype=torch.float)

    return data_input_ids, data_token_type_ids, data_attention_masks, data_scores



def loadData(trainfile,devfile,testfile):
    train=codecs.open(trainfile,'r',encoding='utf-8')
    dev= codecs.open(devfile,'r',encoding='utf-8')
    test = codecs.open(testfile,'r',encoding='utf-8')
    def yielder():
        for lines in (train,dev,test):
            data=[]
            for index, line in enumerate(lines):#.split('\n')):
                line=tuple(line.split('\t'))
                data.append({'index':index, 'sentence_1':line[0],'sentence_2':line[1],'score':float(line[2])})
            yield data
    return tuple([dataset for dataset in yielder()])


class BertRegressionModel(BertPreTrainedModel):
    def __init__(self, bert_model_config: BertConfig):
        super(BertRegressionModel, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        linear_size = bert_model_config.hidden_size
        self.regression = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(linear_size, 1))

    def forward(self, input_ids, token_type_ids, attention_masks):
        _,pooler_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks)
        return self.regression(pooler_output)



class ModelTraining_Evaluation():
    def __init__(self,trainfile,devfile,testfile,bertModel,output_dir):  #bertmodel can be 'bert-base-cased'...,,or any model saved in local computer
        self.trainfile=trainfile
        self.devfile=devfile
        self.testfile=testfile
        self.bertModel=bertModel

        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bertModel, do_lower_case=True)

        self.batch_size = 7
        self.lr=2e-5
        self.eps=1e-8
        self.epochs=1
        self.warmup_steps=2
        self.seed_value=40
        self.output_dir=output_dir


    def getData(self):
        train, dev,test = loadData(self.trainfile, self.devfile, self.testfile)

        train_input_id, train_token_type_id, train_attmask, train_scores = preprocessingForBert(train, self.bert_tokenizer)
        dev_input_id, dev_token_type_id, dev_attmask, dev_scores = preprocessingForBert(dev, self.bert_tokenizer)
        test_input_id, test_token_type_id, test_attmask, test_scores = preprocessingForBert(test, self.bert_tokenizer)

        train_data = TensorDataset(train_input_id, train_token_type_id, train_attmask, train_scores)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        validation_data = TensorDataset(dev_input_id, dev_token_type_id, dev_attmask, dev_scores)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)

        prediction_data = TensorDataset(test_input_id, test_token_type_id, test_attmask, test_scores)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.batch_size)

        return train_dataloader, validation_dataloader, prediction_dataloader,test_input_id


    def training(self):
        model = BertRegressionModel.from_pretrained(self.bertModel)
        model.cuda()
        optimizer = AdamW(model.parameters(), lr=self.lr, eps=self.eps)
        epochs = self.epochs
        warm_up_steps = self.warmup_steps

        train_dataloader, validation_dataloader, prediction_dataloader, test_input_id=ModelTraining_Evaluation.getData(self)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps,num_training_steps=total_steps)

        seed_val = self.seed_value
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        loss_values = []
        loss_function = torch.nn.MSELoss()

        t0 = time.time()
        for epoch_i in range(0, epochs):
            print('======== Epoch {} / {} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            total_loss = 0

            model.train()
            for step, batch in enumerate(train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = time.time() - start_time
                    print('  Batch {}  of  {}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                b_input_ids = batch[0].to(device)
                b_token_type_id = batch[1].to(device)
                b_input_mask = batch[2].to(device)
                b_labels = batch[3].to(device)

                model.zero_grad()
                outputs = model(input_ids=b_input_ids, token_type_ids=b_token_type_id, attention_masks=b_input_mask)
                loss = loss_function(outputs, b_labels)
                loss.backward()
                total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            avg_train_loss = total_loss / len(train_dataloader)
            loss_values.append(avg_train_loss)

            print(" Average training loss: {0:.2f}".format(avg_train_loss))
            print(" Epoch time: {}".format(time.time() - t0))

            print("Running Validation --------")
            model.eval()

            t0 = time.time()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_token_type_id, b_input_mask, b_labels = batch
                with torch.no_grad():
                    outputs = model(b_input_ids, token_type_ids=b_token_type_id, attention_masks=b_input_mask)
                logits = outputs
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

            print("Validation time was : {}".format(time.time() - t0))

        print("Training complete!")

        model_to_save = model.module if hasattr(model,'module') else model
        model_to_save.save_pretrained(self.output_dir)
        self.bert_tokenizer.save_pretrained(self.output_dir)
        return  model


    def evaluation(self):
        train_dataloader, validation_dataloader, prediction_dataloader, test_input_id = ModelTraining_Evaluation.getData(self)
        model = BertRegressionModel.from_pretrained(self.output_dir)
        model.cuda()

        print('Predicting labels for {} sentences'.format(len(test_input_id)))

        model.eval()
        predictions, true_scores = [], []
        for batch in prediction_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_token_type_id,b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=b_token_type_id,attention_masks=b_input_mask)
            logits = outputs
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions.append(logits)
            true_scores.append(label_ids)

        flat_predictions=np.asarray(predictions)
        flat_predictions=flat_predictions.reshape(1,len(test_input_id))

        flat_true_scores=np.asarray(true_scores)
        flat_true_scores=flat_true_scores.reshape(1,len(test_input_id))

        flat_predictions=[item for sublist in flat_predictions for item in sublist]
        flat_true_labels = [item for sublist in flat_true_scores for item in sublist]


        #save results to a csv file
        results_df=pd.DataFrame()
        results_df['true_scores']=np.array(flat_true_labels)
        results_df['predictions']=np.array(flat_predictions)

        print('the pearson correlation score is: {}'.format(results_df.corr(method='pearson')))
        results_df.to_csv('resultsDF.csv',index=False)

        return results_df




train_file = 'STS_data/sts-trainClean.csv'
dev_file='STS_data/sts-devClean.csv'
test_file='STS_data/sts-testClean.csv'

bertModel = 'bert-base-uncased'  #can be changed to other bert models accordingly
output_dir='STS_data/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model=ModelTraining_Evaluation(trainfile=train_file,devfile=dev_file,testfile=test_file,bertModel=bertModel,output_dir=output_dir)


model.training()      #if training model
model.evaluation()     #evaluation using a saved model


print('total time taken: {}'.format(time.time()-start_time))
