import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModel
import os

def model_init(device, pretrained, isFinetune, output):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pretrained == 'base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    elif pretrained=='BlueBERT':
        tokenizer = AutoTokenizer.from_pretrained("../../model/BlueBERT")
        model = AutoModel.from_pretrained("../../model/BlueBERT")
    elif pretrained=='BioBERT':
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
    model.classifier.bias.data[0] = 0.5

    checkpoint = None
    if isFinetune:  # load best model to fine tune
        checkpoint = torch.load(os.path.join(output, "best.pt"))['state_dict']
        for name, param in model.named_parameters():
            if 'module.'+name in checkpoint:
                param = checkpoint['module.'+name]
            param.requires_grad = True
    else: # dont update bert
        for name, param in model.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False
    # model.bert.embeddings.requires_grad = False

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)
    return device, tokenizer, model