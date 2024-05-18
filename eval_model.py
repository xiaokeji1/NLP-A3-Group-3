import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from sa_model import SAModel

class TextDataset(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas               
        self.labels = labels           
        self.tokenizer = BertTokenizer.from_pretrained('/root/.cache/models/bert-base-uncased')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        text = self.datas[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # 添加特殊的'[CLS]'和'[SEP]'
            max_length=512,  # 设置最大序列长度
            padding='max_length',  # 进行填充
            return_attention_mask=True,  # 返回注意力掩码
            return_tensors='pt',  # 返回PyTorch张量
            truncation=True  # 进行截断
        )
        labels = self.labels[idx]
        return {
            'input_ids': encoding['input_ids'].flatten(),            # 输入id
            'attention_mask': encoding['attention_mask'].flatten(),  # 注意力掩码
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# 测试数据集
val_data = np.load('/root/sentiment_analysisB/process_data/val_text.npy')
val_label = np.load('/root/sentiment_analysisB/process_data/val_label.npy')
print(val_data.shape, val_label.shape)
val_dataset = TextDataset(val_data, val_label)
val_dataloader = DataLoader(val_dataset, batch_size = 16)

# 初始化Transformer模型
model = SAModel()
model_weights = torch.load('model_weights.pth')  
model.load_state_dict(model_weights)  
if torch.cuda.is_available():  
    model.cuda()
    print("load on gpu")

def calculate_metrics(TP, TN, FP, FN):
    # 计算准确率
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # 计算召回率
    recall = TP / (TP + FN)

    # 计算精确率
    precision = TP / (TP + FP)

    # 计算F1分数
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, recall, precision, f1_score

def eval_model(test_size):
    matrix = torch.zeros([2, 2], dtype=torch.long).to('cuda')                  # 混淆矩阵  
    total, total_correct = 0, 0
    for i, batch in enumerate(val_dataloader):
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')
        logits = model(input_ids, attention_mask)         


        _, predicted = torch.max(logits, 1)
        correct = (predicted == labels).float()  
        for i, pred in enumerate(predicted):
            matrix[pred.item()][labels[i].item()] += 1            
        # 计算准确率（正确的预测数 / 总样本数）  
        acc = correct.sum() / len(labels)
        print(f"ACC: {acc}")
        
        total += len(labels)
        total_correct += correct.sum()
        if total > test_size: 
            break
    tp, tn, fp, fn = matrix[1][1], matrix[0][0], matrix[1][0], matrix[0][1]
    accuracy, recall, precision, f1 = calculate_metrics(tp, tn, fp, fn)
    acc = total_correct / total
    print(f"ACC: {acc}")
    print("accuracy: {}. recall: {}. precision: {}. f1: {}.".format(accuracy, recall, precision, f1))

eval_model(test_size = float('inf'))
