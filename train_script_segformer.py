import os

from datasets import load_dataset
from datasets import Dataset
import numpy as np
from PIL import Image
from torchvision.transforms import ColorJitter
from transformers import SegformerFeatureExtractor
import json

data_split = ['train','val']

path = '/scratch/j/jcaunedo/umar1/segmentation/IDD_Segmentation/'

fl = open('label2id.json')
label2id = json.load(fl)
fl.close()

fl = open('id2label.json')
id2label = json.load(fl)
fl.close()

def populate(cpath):
    
    result = []
    
    folders = os.listdir(cpath)
    
    for each_folder in folders:
        
        combined_folder_path = cpath+'/'+each_folder

        files = os.listdir(combined_folder_path)
        
        for each in files:
            
            result.append(combined_folder_path+'/'+each)
    
    return result
    
def get_all_images_path(split,subset=0):
    
    combined_path = path+'leftImg8bit/'+split+'/'
    
    images = sorted(populate(combined_path))
    
    combined_path = path+'label_processed/'+split+'/'
    
    masks = sorted(populate(combined_path))
    
    if subset>0:
        images = images[:subset]
        masks = masks[:subset]
    
    data_dict = {
        "pixel_values" : images,
        "label" : masks
    }
    
    return data_dict
    

    
d_train =  get_all_images_path('train')
d_val = get_all_images_path('val')
# Create a Hugging Face Dataset from the dictionary
dataset_train = Dataset.from_dict(d_train)
dataset_val = Dataset.from_dict(d_val)


def mapping_fn(example):
    result = {}
    result['pixel_values'] = Image.open(example['pixel_values']).convert('RGB')
    result['label'] = Image.open(example['label']).convert('L')
    return result
    


feature_extractor = SegformerFeatureExtractor()
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 

def train_transforms(example_batch):
    images = [jitter(Image.open(x).convert('RGB')) for x in example_batch['pixel_values']]
    labels = [Image.open(x).convert('L') for x in example_batch['label']]
    inputs = feature_extractor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [Image.open(x).convert('RGB') for x in example_batch['pixel_values']]
    labels = [Image.open(x).convert('L') for x in example_batch['label']]
    inputs = feature_extractor(images, labels)
    return inputs


# Set transforms
dataset_train.set_transform(train_transforms)
dataset_val.set_transform(val_transforms)


from transformers import SegformerForSemanticSegmentation

pretrained_model_name = "nvidia/mit-b0" 
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id
)

from transformers import TrainingArguments

epochs = 50
lr = 0.00006
batch_size = 16
.
.


training_args = TrainingArguments(
    "segformer-b0-finetuned-segments-sidewalk-outputs",
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_steps=1,
    eval_accumulation_steps=5,
    load_best_model_at_end=True

)


import torch
from torch import nn
import evaluate

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
  with torch.no_grad():
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    # scale the logits to the size of the label
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    pred_labels = logits_tensor.detach().cpu().numpy()
    # currently using _compute instead of compute
    # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
    metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=feature_extractor.do_reduce_labels,
        )
    
    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[str(i)]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[str(i)]}": v for i, v in enumerate(per_category_iou)})
    
    return metrics

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    compute_metrics=compute_metrics,
)

trainer.train()