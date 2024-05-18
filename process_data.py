
import numpy as np
import re
# Read the file
with open("/root/sentiment_analysisB/process_data/data.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
# Define regular expression pattern
pattern = r"__label__(\d+)\s+(.+)"

# Data processing
data = []
m, x, y, total = 0, 0, 0, 0
for line in lines:
    match = re.match(pattern, line)
    if match:
        label = match.group(1)
        text = match.group(2)
        data.append((int(label)-1, text))
        m = max(m, len(text))
        if len(text) > 512:         #
            x += 1
            if len(text) > 768:
                y += 1
        total += 1
    if total > 15000: break
print(m, x, y, total)
import time
# time.sleep(150000)
# Shuffle the data
np.random.seed(42)
np.random.shuffle(data)

# Divide the dataset
total_samples = len(data)
train_size = int(0.6 * total_samples)
val_size = int(0.2 * total_samples)

train_data = np.array(data[:train_size])
val_data = np.array(data[train_size:train_size+val_size])
test_data = np.array(data[train_size+val_size:])
print("examine train data", train_data.shape, type(train_data))

# Separate label and text
train_label, train_text = train_data[:, 0], train_data[:, 1]
val_label, val_text = val_data[:, 0], val_data[:, 1]
test_label, test_text = test_data[:, 0], test_data[:, 1]
print("examine train label, train text", train_label.shape, train_text.shape, type(train_label), type(train_text))
print("example train label dtype", type(train_label[0]))
train_label, val_label, test_label = train_label.astype(int), val_label.astype(int), test_label.astype(int)
print("example train label dtype", type(train_label[0]))


# Display data size
print("训练集大小:", len(train_data))
print("验证集大小:", len(val_data))
print("测试集大小:", len(test_data))

# Save the selected elements as a new .npy file 
np.save('/root/sentiment_analysisB/process_data/train_text.npy', train_text)  
np.save('/root/sentiment_analysisB/process_data/train_label.npy', train_label)  
np.save('/root/sentiment_analysisB/process_data/val_text.npy', val_text)  
np.save('/root/sentiment_analysisB/process_data/val_label.npy', val_label)  
np.save('/root/sentiment_analysisB/process_data/test_text.npy', test_text)  
np.save('/root/sentiment_analysisB/process_data/test_label.npy', test_label)  
