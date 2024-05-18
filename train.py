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


# 训练数据集
train_data = np.load('/root/sentiment_analysisB/process_data/train_text.npy')
train_label = np.load('/root/sentiment_analysisB/process_data/train_label.npy')
print(train_data.shape, train_label.shape)
dataset = TextDataset(train_data, train_label)
dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)

# 测试数据集
val_data = np.load('/root/sentiment_analysisB/process_data/val_text.npy')
val_label = np.load('/root/sentiment_analysisB/process_data/val_label.npy')
print(val_data[0], val_label[0])
print(val_data.shape, val_label.shape)
val_dataset = TextDataset(val_data, val_label)
val_dataloader = DataLoader(val_dataset, batch_size = 16, shuffle = False)


# 初始化 Transformer 模型
model = SAModel()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# decayRate = 0.96
# my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=decayRate)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 训练模型
model.train()
if torch.cuda.is_available():  
    model.cuda()
    print("load on gpu")


def eval_model(test_size):
    total, total_correct = 0, 0
    for i, batch in enumerate(val_dataloader):
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')
        logits = model(input_ids, attention_mask)         
    
        _, predicted = torch.max(logits, 1)
        correct = (predicted == labels).float()  
            
        # 计算准确率（正确的预测数 / 总样本数）  
        acc = correct.sum() / len(labels)
        print(f"ACC: {acc}")
        
        total += len(labels)
        total_correct += correct.sum()
        if total > test_size: 
            break
         
    acc = total_correct / total
    print(f"Total ACC: {acc}")
    
import matplotlib.pyplot as plt  
losses = []
cnt = 0
for epoch in range(100000000000000000000):  
    for batch in dataloader:
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')
        # print("---", inputs.shape, labels.shape)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)             # (Batch, )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if cnt % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}", cnt)
        
        if cnt % 100 == 0:
            torch.save(model.state_dict(), 'model_weights_test.pth')
            eval_model(test_size = 64)
            losses.append(loss.item())  
            # 在训练结束后绘制损失曲线图  
            plt.plot(losses)  
            plt.xlabel('Epoch')  # 如果你想按epoch显示，可以修改为'Epoch'并使用epoch // len(dataloader)作为x轴值  
            plt.ylabel('Loss')  
            plt.title('Loss Curve')  
      
            plt.savefig('loss_curve.jpg')
        cnt += 1

        if loss.item() < 0.01:
            break
    