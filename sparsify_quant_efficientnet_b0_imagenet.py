import os
import sys
import numpy as np
import torch
import torch.nn as nn
import timm
from utils import generate_exp, get_layer_by_name, validate, calculate_sparsity
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
import onnx
from onnxsim import simplify
from sparseml.pytorch.optim import ScheduledModifierManager
import math
from sensitivity import sensitivity_analysis


def export_onnx(model, export_path:str, img_size=(224,224)):
    # Export the model
    x = torch.zeros((1,3, img_size[0], img_size[1]), requires_grad=True)
    torch.onnx.export(
                    model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    export_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=14,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
    )
    # convert model
    original_model = onnx.load(export_path)
    model_simp, check = simplify(original_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save_model(model_simp, export_path)


if __name__ == "__main__":
    ROOT_EXP = "runs"
    BATCHSIZE = 32
    VAL_BATCHSIZE = 256
    VAL_ITER = 2

    exp_name = generate_exp(ROOT_EXP)
    writer = SummaryWriter(exp_name)
    os.makedirs(exp_name, exist_ok=True)
    exp_export_path = os.path.join(exp_name, "export")
    os.makedirs(exp_export_path, exist_ok=True)

    # model setup:
    model = timm.create_model('tf_efficientnet_b0.in1k', pretrained=True, exportable=True)
    export_onnx(model, export_path=os.path.join(exp_export_path, "efficientnet-b0-imagenet-base.onnx"))

    # setup dataset:
    data_config = timm.data.resolve_model_data_config(model)
    train_transforms = timm.data.create_transform(**data_config, is_training=True)    
    val_transforms = timm.data.create_transform(**data_config, is_training=False)    

    def preproc_train_transforms(examples):
        examples['image'] = [ train_transforms(image.convert('RGB')) for image in examples["image"]] 
        return examples
    
    def preproc_val_transforms(examples):
        examples['image'] = [ val_transforms(image.convert('RGB')) for image in examples["image"]] 
        return examples

    """
    def preproc_train_transforms(examples):
        examples['image'] = train_transforms(examples['image'].convert('RGB')) 
        return examples
    
    def preproc_val_transforms(examples):
        examples['image'] = val_transforms(examples['image'].convert('RGB')) 
        return examples
    """ 
    train_dataset = load_dataset("imagenet-1k", split="train", streaming=False, trust_remote_code=True)
    val_dataset = load_dataset("imagenet-1k", split="validation", streaming=False, trust_remote_code=True)
    #val_dataset = load_dataset("timm/imagenet-1k-wds", split="validation")

    train_dataset.set_transform(preproc_train_transforms)
    val_dataset.set_transform(preproc_val_transforms)
    #train_dataset = train_dataset.map(preproc_train_transforms)
    #val_dataset = val_dataset.map(preproc_val_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4) 
    val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCHSIZE, shuffle=False, num_workers=4)
    #train_length = math.ceil(1281167/BATCHSIZE)
    train_length = len(train_dataloader)

    if torch.cuda.is_available():
        model = model.cuda()

    # sensitivity testing (disable when you have the layers you want):
    #sensitivity_analysis(model, val_dataloader, exp_name, None)

    # training hyp
    PATH_TO_RECIPE = "efficientnet_b0_layer.yaml"
    LR = 0.01
    
    manager = ScheduledModifierManager.from_yaml(PATH_TO_RECIPE)
    EPOCHS = manager.max_epochs
    FINETUNE_EPOCH_1 = int(EPOCHS*0.8)
    FINETUNE_EPOCH_2 = int(EPOCHS*0.9)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [FINETUNE_EPOCH_1, FINETUNE_EPOCH_2], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    optimizer = manager.modify(model, optimizer, train_length)
    scaler = torch.amp.GradScaler("cuda")

    BEST_LOSS = 1e100
    BEST_VAL = 0.0

    acc = validate(model, val_dataloader, override_to_cpu=False, set_model_mode=True)
    writer.add_scalar("original_acc", acc, 0)
    running_items = train_length

    for epoch in range(EPOCHS):
        logger.info("Epoch: {}".format(epoch))
        running_loss = 0.0
        #train_dataset.shuffle()

        for data in tqdm(train_dataloader):
            optimizer.zero_grad(set_to_none=True)

            images = data["image"]
            labels = data["label"]
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            with torch.autocast(device_type="cuda:0", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            #loss.backward()
            scaler.step(optimizer)
            #optimizer.step()
            scaler.update()
            running_loss += loss.item()

        scheduler.step(epoch)
        _loss = running_loss/running_items
        logger.info("Loss: {}".format(_loss))

        lr = optimizer.param_groups[0]['lr']
        logger.debug("\tLoss: {}".format(lr))

        if _loss < BEST_LOSS:
            logger.success("Best loss: {}".format(_loss))
            BEST_LOSS = _loss
            # save model here:
            torch.save(model.state_dict(), os.path.join(exp_name, "best_loss.pt"))
        writer.add_scalar("train/loss", _loss, epoch)

        if epoch % VAL_ITER == 0:
            acc = validate(model, val_dataloader)
            logger.success("Best val acc: {}".format(acc))
            writer.add_scalar("val/acc", acc, epoch)

            if acc > BEST_VAL:
                BEST_VAL = acc
                # save model here:
                torch.save(model.state_dict(), os.path.join(exp_name, "best_val.pt"))

    manager.finalize(model)
    acc = validate(model, val_dataloader)

    model = model.cpu()
    sparsity_level = calculate_sparsity(model)
    print("> Final validation score: {}".format(acc))
    print("> Final sparsity level: {}".format(sparsity_level))
    torch.save(model.state_dict(), os.path.join(exp_name, "finalize_model.pt"))

    export_onnx(model, export_path=os.path.join(exp_export_path, "efficientnet-b0-imagenet-pruned.onnx"))