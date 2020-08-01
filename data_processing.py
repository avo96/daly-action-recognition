import pickle
import pandas as pd
import numpy as np

from model import torch,DataLoader,Dataset
from image_utils import *

def parse_annots(annots_file,width=1280,height=720):
    with open(annots_file, "rb") as f:
        daly = pickle.load(f, encoding='latin1')
    rows=[]
    columns=['photo_path','label','xmin','ymin','xmax','ymax']
    for vid in daly['annot'].keys():
        action_list=list(daly['annot'][vid]['annot'].keys())
        for action in action_list:
            act=daly['annot'][vid]['annot'][action]
            for i in range(len(act)):
                for j in range(len(act[i]['keyframes'])):
                    time=str(act[i]['keyframes'][j]['time'])
                    if len(time.split('.')[1])==1:
                        time=time+'0'
                    path='daly_images/'+vid+'/img_'+time+'.jpg'
                    bbox=np.array(*act[i]['keyframes'][j]['boundingBox'],dtype=np.float32)
                    rows.append([path,action,width*bbox[0],height*bbox[1],width*bbox[2],height*bbox[3]])
    return pd.DataFrame(rows,columns=columns)


class DALY(torch.utils.data.Dataset):
    def __init__(self, images_path, labels, boxes):
        self.images_path = images_path
        self.labels = labels
        self.boxes = boxes

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        path = self.images_path[idx]
        label = self.labels[idx]
        x, y_bb = transformsXY(path, self.boxes[idx])
        x = normalize(x)
        x = np.rollaxis(x, 2)
        return x, label, y_bb

