import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModel
import os
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, bert):
        super(Identity, self).__init__()
        self.encoder = bert

    def forward(self, input_ids, attention_mask = None):
        return self.encoder(input_ids, attention_mask=attention_mask).logits

    def get_scores(self, inputs):
        x = self.encoder(**inputs).logits
        scores = torch.softmax(x, dim=1)[:, 1]
        return scores

class SimpleModel(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(768, 2) # for base BERT

    def forward(self, input_ids, attention_mask = None):
        x = self.bert(input_ids, attention_mask=attention_mask)[1]
        return self.classifier(x)

    def get_scores(self, inputs):
        x = self.bert(inputs)[1]
        logits = self.classifier(x)
        scores = torch.softmax(logits, dim=1)[:, 1]

        return scores


def model_init(device, pretrained, isFinetune, output, path_to_trained_model):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pretrained == 'base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        bert.classifier = torch.nn.Linear(bert.classifier.in_features, 2)
        model = Identity(bert)
    elif pretrained == 'BlueBERT':
        tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
        bert = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
        model = SimpleModel(bert)
    elif pretrained == 'BioBERT':
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        bert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        model = SimpleModel(bert)
    elif pretrained == 'ClinicalBERT':
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model = SimpleModel(bert)
    elif pretrained == 'SciBERT':
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
        bert = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")
        model = SimpleModel(bert)
    else:
        raise Exception("Not matched pretrained model")

    if path_to_trained_model:
        try:
            checkpoint = torch.load(os.path.join(path_to_trained_model, "best.pt"))['state_dict']
            model.load_state_dict(checkpoint)
        except:
            raise Exception("{} pretrained key not match".format(pretrained))

    # if isFinetune:  # load best model to fine tune
    #     for name, param in model.named_parameters():
    #         param.requires_grad = True
    #     #     if 'module.'+name in checkpoint :
    #     #         param = checkpoint['module.'+name]
    #     #     else:
    #     #         print('name {} not in checkpoint', name)
    # else: # dont update bert
    #     for name, param in model.named_parameters():
    #         if not name.startswith('classifier'):
    #             param.requires_grad = False
    # # model.bert.embeddings.requires_grad = False

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)
    return device, tokenizer, model
