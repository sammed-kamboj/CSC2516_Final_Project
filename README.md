# CSC2516_Final_Project
Semantic segmentation of driving scenes. 

1. Install requirements using requirements.txt

2. Download the data from the website https://idd.insaan.iiit.ac.in/dataset/download/

3. Run id2label.ipynb - This will ensure consistent label to id conversion and it will generate two files id2label.json and label2id.json which will be used later on.

4. Run preprocess.ipynb - This will preprocess the json label files from the dataset. It will convert the polygon and object information from the json files into label images. 

5. train_script_segformer.py - This script will train the segformer model on the preprocessed data
