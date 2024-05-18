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
            add_special_tokens=True, # Add special '[CLS]' and '[SEP]'
            max_length=512, # Set the maximum sequence length
            padding='max_length', # Fill
            return_attention_mask=True,  # Return the attention mask
            return_tensors='pt',  # Return a PyTorch tensor
            truncation=True  # Truncate
        )
        labels = self.labels[idx]
        return {
            'input_ids': encoding['input_ids'].flatten(),           # Input id
            'attention_mask': encoding['attention_mask'].flatten(),  # Attention Mask
            'labels': torch.tensor(labels, dtype=torch.long)
        }


# Training dataset
train_data = np.load('/root/sentiment_analysisB/process_data/train_text.npy')
train_label = np.load('/root/sentiment_analysisB/process_data/train_label.npy')
print(train_data.shape, train_label.shape)
dataset = TextDataset(train_data, train_label)
dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)

# Test dataset
val_data = np.load('/root/sentiment_analysisB/process_data/val_text.npy')
val_label = np.load('/root/sentiment_analysisB/process_data/val_label.npy')
print(val_data[0], val_label[0])
print(val_data.shape, val_label.shape)
val_dataset = TextDataset(val_data, val_label)
val_dataloader = DataLoader(val_dataset, batch_size = 16, shuffle = False)


# Initialize the Transformer model
model = SAModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
# decayRate = 0.96
# my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=decayRate)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Train the model
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
            
       # Calculate accuracy (number of correct predictions / total number of samples)
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
            # Draw the loss curve after training 
            plt.plot(losses)  
            plt.xlabel('Epoch')  # If you want to display by epoch, you can change it to 'Epoch' and use epoch // len(dataloader) as the x-axis value  
            plt.ylabel('Loss')  
            plt.title('Loss Curve')  
      
            plt.savefig('loss_curve.jpg')
        cnt += 1

        if loss.item() < 0.01:
            break
    
