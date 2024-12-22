import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn   
from copy import deepcopy
from utils import validate, get_layer_by_name
from loguru import logger
from tqdm import tqdm

def random_weight_masking(layer, model, sparsity:float, seed:int):
    np.random.seed(seed)
    _layer = get_layer_by_name(model, layer)
    _weights = _layer.weight
    _weight_mask = np.random.uniform(0,1,size=_weights.shape)
    _weight_mask = (_weight_mask > sparsity).astype(float)
    _weight_mask = torch.Tensor(_weight_mask).to(_layer.weight.device)
    #_w = np.count_nonzero(_weight_mask)
    #print(_w, _w/np.prod(_weight_mask.shape) )
    _layer.weight *= _weight_mask

def sensitivity_analysis(model, dataloader, export_path:str, layers):
    model.eval()

    # get number of weights per layer:
    if layers is None:
        plt.figure(figsize=(10,5))
        layer_labels = []
        layer_num = []
        for name, param in model.named_parameters():
            #if "weight" in name and not "bn" in name:
            if "weight" in name:
                _root = name.replace(".weight", "")
                _num_weights = np.prod(param.shape)
                if _num_weights > 0:
                    print(_root, _num_weights)
                    layer_labels.append(_root)
                    layer_num.append(_num_weights)

        biggest_layers = [[x,y] for y, x in sorted(zip(layer_num, layer_labels), key=lambda pair: pair[0], reverse=True)]
        name_layers = [x for x,_ in biggest_layers]
        num_neurons = [y for _,y in biggest_layers]

        with open(os.path.join(export_path, "layer_list.txt"), "w") as f:
            f.write("name,num_param\n")
            for i in range(len(name_layers)):
                f.write("{},{}\n".format(name_layers[i], num_neurons[i]))

        best_k = 10
        name_layers = name_layers[:best_k]
        num_neurons = num_neurons[:best_k]
        layers = name_layers

        y_layer_pos = [i for i in name_layers]
        plt.barh(y_layer_pos, num_neurons)
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(export_path, "layer_ranking.svg"))
        plt.close()

    logger.info("Analyzing {}".format(layers))

    _model_data = deepcopy(model.state_dict())
    sparse_list = [i/100 for i in range(10,100,10)]
    layer_val_acc = {_layer:{} for _layer in layers}
    iter_num = 1
    val_acc = validate(model, dataloader)

    with torch.no_grad():
        # modify network layer by layer:
        for layer in layers:
            layer_val_acc[layer][0] = val_acc

        for sparsity in tqdm(sparse_list, desc="Layer Progress"):
            for layer in layers:
                layer_val_acc[layer][sparsity] = []

                for i in range(iter_num):
                    random_weight_masking(layer, model, sparsity, i)
                    val_acc = validate(model, dataloader)
                    layer_val_acc[layer][sparsity].append(val_acc)

                    # reset model weights:
                    model.load_state_dict(_model_data)
    
    # graphing:
    plt.figure()

    all_data = {}
    for layer in layers:
        # list that contains all sparsity levels 
        averaged_vals = [] 
        max_val = [] 
        min_val = []
        
        for sparsity in [0]+sparse_list:
            _vals = layer_val_acc[layer][sparsity]
            _mean = np.mean(_vals)
            _max = np.max(_vals)
            _min = np.min(_vals)
            averaged_vals.append(_mean)
            max_val.append(_max)
            min_val.append(_min) 
        
        plt.plot([0]+sparse_list, averaged_vals, label=layer)
        all_data[layer] = [averaged_vals, max_val, min_val]   

    plt.legend()
    plt.savefig(os.path.join(export_path, "sensitivity.svg"))
    #plt.show()
    np.save(os.path.join(export_path, "raw_sensitivity.npy"), all_data)
    model.train()

if __name__ == "__main__":
    pass