import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score

def validate(model, val_loader, set_model_mode=True, override_to_cpu=False):
    if set_model_mode:
        model.eval()

    all_preds = torch.zeros((0,))
    all_labels = torch.zeros((0,))

    with torch.no_grad():
        if torch.cuda.is_available() and not override_to_cpu:
            all_preds = all_preds.cuda()
            all_labels = all_labels.cuda()

        for data in tqdm(val_loader, desc="Validation"):
            images = data["image"]
            labels = data["label"]
            if torch.cuda.is_available() and not override_to_cpu:
                images = images.cuda()
                labels = labels.cuda()
            logits = model(images)
            scores = nn.functional.softmax(logits, dim=1)
            _cls, _cls_arg = torch.max(scores, dim=1)

            #if all_preds is None:
            #    all_preds = torch.clone(_cls_arg)    
            #    all_labels = torch.clone(labels)
            #else:
            all_preds = torch.concat((all_preds, _cls_arg))
            all_labels = torch.concat((all_labels, labels))

    all_labels = all_labels.cpu().numpy()
    all_preds = all_preds.cpu().numpy()
    conf_mat = confusion_matrix(all_labels, all_preds)
    acc_per_class = conf_mat.diagonal()/conf_mat.sum(axis=1)
    acc = accuracy_score(all_labels, all_preds)

    if set_model_mode:
        model.train()
    return acc

def generate_exp(root:str, template="run"):
    iter = 0 
    _full_folder_path = None
    while True:
        _folder_attempt = "{}_{}".format(template, iter)
        _full_folder_path = os.path.join(root, _folder_attempt)
        if not os.path.exists(_full_folder_path):
            break
        iter += 1
    return _full_folder_path

# Function to access nested layers using a string
def get_layer_by_name(model, layer_name):
    layers = layer_name.split('.')
    layer = model
    for l in layers:
        if '[' in l and ']' in l:
            # Handle indexing in list or dictionary
            layer_name, index = l.split('[')
            index = int(index[:-1])  # Convert index to integer
            layer = getattr(layer, layer_name)[index]
        else:
            layer = getattr(layer, l)
    return layer

def calculate_sparsity(model):
    total_params = 0
    zeros_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zeros_params += torch.sum(param == 0).item()
    sparsity = zeros_params/total_params
    return sparsity