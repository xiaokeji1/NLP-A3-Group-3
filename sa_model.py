
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple
from transformers import BertModel
from ding_transformer import *
from ding_module import fc_block, build_normalization




class SAModel(nn.Module):
    """
    Overview:
        Implementation of the Transformer model.

    .. note::
        For more details, refer to "Attention is All You Need": http://arxiv.org/abs/1706.03762.

    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        head_dim: int = 128,
        hidden_dim: int = 1024,
        output_dim: int = 256,
        head_num: int = 2,
        mlp_num: int = 2,
        layer_num: int = 3,
        dropout_ratio: float = 0.,
        activation: nn.Module = nn.ReLU(),
    ):
        """
        Overview:
            Initialize the Transformer with the provided dimensions, dropout layer, activation function,
            and layer numbers.
        Arguments:
            - input_dim (:obj:`int`): The dimension of the input.
            - head_dim (:obj:`int`): The dimension of each head in the multi-head attention mechanism.
            - hidden_dim (:obj:`int`): The dimension of the hidden layer in the MLP (Multi-Layer Perceptron).
            - output_dim (:obj:`int`): The dimension of the output.
            - head_num (:obj:`int`): The number of heads in the multi-head attention mechanism.
            - mlp_num (:obj:`int`): The number of layers in the MLP.
            - layer_num (:obj:`int`): The number of Transformer layers.
            - dropout_ratio (:obj:`float`): The dropout ratio for the dropout layer.
            - activation (:obj:`nn.Module`): The activation function used in the MLP.
        """
        super(SAModel, self).__init__()
        self.bert_model = BertModel.from_pretrained('/root/.cache/models/bert-base-uncased')
        for _, param in self.bert_model.named_parameters():
            param.requires_grad = False
        self.decoder = fc_block(768, 2, activation=activation)      


    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Overview:
            Perform the forward pass through the Transformer.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor, with shape `(B, N)`, 
                where `B` is batch size, \
                      `N` is the number of entries.      
            - mask (:obj:`Optional[torch.Tensor]`, optional): The mask tensor (bool), used to mask out invalid \
                entries in attention. It has shape `(B, N)`, where `B` is batch size and `N` is number of \
                entries. Defaults to None.
        Returns:
            - x (:obj:`torch.Tensor`): The output tensor from the Transformer.
        """
        # 获取BERT模型的输出
        llm_output = self.bert_model(x, mask)
        # 输出将包含两个主要部分：'last_hidden_state' 和 'pooler_output'
        last_hidden_states = llm_output.last_hidden_state
        pooler_output = llm_output.pooler_output
        # print("examine output:", pooler_output.shape, type(pooler_output), pooler_output.device)
        logits = self.decoder(pooler_output)                                                               # (B, 60)
        return logits 

