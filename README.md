# NLP-A3-Group-3
# Sentiment analysis of product reviews based on BERT model


## **Operating environment：**

torch==1.8.0
python==3.8.12



## data set：

Product review original data set data.txt
Label 0 and 1 binary classification


## Module description
### Data processing module:
process_data.py 
Divide the data into training, verification and testing. The ratio is 3:1:1, of which 9000 are trained, 3000 are verified, and 3000 are tested.
The processed data is saved in the process_data folder, and the files are as follows:
train_text.npy
train_label.npy
val_text.npy
val_label.npy
test_text.npy
test_label.npy
### Model file module:
#### ding_transformer.py
A Transformer model is defined, which includes the Attention module and TransformerLayer module of the self-attention mechanism. Here's an explanation of the code:

The Attention class defines an attention mechanism module for calculating attention weight and attention output. It accepts input dimension input_dim, head dimension head_dim, output dimension output_dim, head number head_num and dropout module. attention_pre is a fully connected layer used to linearly map input features into queries, keys and values. The split method splits and transposes the input tensor according to the number of heads. The forward method calculates the attention weight, attention output and returns the result.
The TransformerLayer class defines a Transformer layer, including self-attention mechanism and multi-layer perceptron. It accepts the input dimension input_dim, the head dimension head_dim, the hidden dimension hidden_dim, the output dimension output_dim, the number of heads head_num, the number of layers of the multi-layer perceptron mlp_num, the dropout module and the activation function. The forward method accepts input tensors and mask tensors, calculates self-attention output, multi-layer perceptron output and returns the results.
The Transformer class defines a complete Transformer model, including an embedding layer and multiple Transformer layers. It accepts the input dimension input_dim, the head dimension head_dim, the hidden dimension hidden_dim, the output dimension output_dim, the number of heads head_num, the number of layers of the multi-layer perceptron mlp_num, the number of Transformer layers layer_num, the dropout ratio dropout_ratio and the activation function. The forward method accepts input tensors and mask tensors, calculates the output of the embedding layer, the output of multiple Transformer layers and returns the results.
The ScaledDotProductAttention class defines a module based on the scaled dot product attention mechanism. It accepts input query tensor q, key tensor k, value tensor v and mask tensor mask, calculates attention weights and returns attention output.
#### ding_module.py
Several utility functions and modules for neural networks are defined:
weight_init_: Initialize the weight of the tensor using Xavier, Kaiming or orthogonal initialization methods.
sequential_pack: Convert a sequence of layers into sequential modules.
fc_block: Constructs a fully connected block with optional activation, normalization and dropout layers.
normed_linear: Build a linear layer with normalized weights.
normed_conv2d: Constructs a 2D convolutional layer with normalized weights.
MLP: Constructs a multi-layer perceptron with fully connected layers, optional activation, normalization and dropout layers.
ChannelShuffle: Shuffles the channels in the tensor.
one_hot: Convert integer values to one-hot encoded tensors.
NearestUpsample: Performs nearest neighbor upsampling on a tensor.

#### ding_norm.py
This code is used to build the corresponding normalization module. According to the passed normalization type and dimension, the corresponding normalization function is returned. The currently supported normalization types are ['BN', 'LN', 'IN', 'SyncBN'], where BN represents batch normalization, LN represents layer normalization, IN represents instance normalization, and SyncBN represents Synchronous batch normalization. When the normalization type is BN or IN, the dimension dim needs to be specified. The returned normalization function is the corresponding batch normalization module.


#### sa_model.py
This model is an implementation of the Transformer model. It uses the pre-trained BERT model as input and builds a Transformer structure with a multi-layer self-attention mechanism on it. The input to the model is a tensor of shape (B, N), where B is the batch size and N is the length of the input. The model obtains the input representation through the BERT model, and then maps the representation to the output dimensions through a fully connected block. The final output is a tensor of shape (B, 2), where 2 represents the number of output categories of the model. The goal of this model is to perform classification tasks on inputs.

### Model training module
train.py
This code is a complete training process for training a sentiment analysis model. First, a `TextDataset` class is defined for loading data, and `BertTokenizer` is used to process the text. Then load the training data set and test data set into `DataLoader`. Next, a `SAModel` model is initialized, and the loss function `CrossEntropyLoss` and the optimizer `AdamW` are defined. Then enter the training loop, obtain the data of each batch by traversing the `dataloader`, and pass it into the model for forward calculation. The calculated logits and labels are used to calculate cross-entropy loss, and backpropagation and parameter updating are performed. At the beginning of each epoch, the current model weights are saved and the `eval_model` function is called to evaluate the model. Finally, save the loss value and draw the loss curve. The entire training process will continue until the loss value is less than 0.01 or the set maximum number of epochs is reached.

### Model evaluation module
eval_model.py
This code is used to evaluate the performance of the trained model on the test set. First, load the test data set and process the data through the TextDataset class. Then, initialize a SAModel model and load the model parameters from the saved weight file. Next, the test set is traversed and the data is passed into the model for forward calculation. The calculated logits are compared with the labels to obtain the prediction results. At the same time, based on the prediction results and real labels, each indicator in the confusion matrix (true positive, true negative, false positive, false negative) is calculated. Finally, based on the indicators in the confusion matrix, calculate the accuracy, recall, precision and F1 scores, and print out the values of these indicators. This completes the evaluation process of the model on the test set.
