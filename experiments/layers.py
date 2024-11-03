import math, copy, os.path
import torch
from typing import Union


'''
Custom attention layers
'''


class BertAttention(torch.nn.Module):
    '''
    Definition of BERT attention layer is as follows:

        out = softmax( (Q * K^T) / sqrt(d)) * V

    where Q, V and K are linear transforms of the input
    resp. activation of previous layer.

    Derived from 
        
        https://stackoverflow.com/questions/76648620/how-do-i-implement-this-attention-layer-in-pytorch
    
    '''
    def __init__(self, input_dim: int) -> None:
        '''
        Constructor
        '''
        super(BertAttention, self).__init__()
        self.feature_size = input_dim
        self.query = torch.nn.Linear(input_dim, input_dim)
        self.key = torch.nn.Linear(input_dim, input_dim)
        self.value = torch.nn.Linear(input_dim, input_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''
        Init parameters with random weights
        '''
        stdv = 1.0 / math.sqrt(self.feature_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input: torch.Tensor, mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for back-propagation
        '''
        # Apply linear transformations
        keys = self.key(input)
        queries = self.query(input)
        values = self.value(input)
        # Scaled dot-product attention
        scores = torch.matmul(queries.transpose(0,1), keys) / (self.feature_size**0.5)
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # Apply softmax
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        # Multiply weights with values
        out = torch.matmul(values, attention_weights)
        return out, attention_weights
    

class SelfAttention(torch.nn.Module):
    '''
    Class implementing self-attention from 
        
        https://arxiv.org/pdf/1512.08756.pdf
    
    Works in principle for RNN layers. The definitions are as follows:

        E_t = a(H_t)
        Alpha_t = softmax(E_t)
        C = \sum_{t} Alpha_t * H_t
    
    '''
    def __init__(self, input_dim:int) -> None:
        '''
        Constructor
        '''
        super(SelfAttention,self).__init__()
        self.hidden = torch.nn.Linear(input_dim[-1], 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''
        Init parameters with random weights
        '''
        stdv = 1.0 / math.sqrt(self.feature_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)      
 
    def forward(self, input:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for back-propagation
        '''
        # Alignment scores. Pass them through tanh function
        e = torch.tanh(torch.dot(input, self.self.hidden)) 
        # Compute the weights
        alpha = torch.softmax(e.squeeze(-1))
        alpha = alpha.unsqueeze(-1)
        # Compute the context vector
        context = input * alpha
        output = torch.nn.functional.sum(context, dim=1)
        return context, output


'''
Model architectures
'''


class ModLogReg(torch.nn.Module):
    '''
    Logistic regression in Pytorch (single layer followed by softmax)
    '''
    def __init__(self, input_features, 
                 out_features):

        super(ModLogReg, self).__init__()
        self.linear = torch.nn.Linear(input_features, out_features)
        self.output_layer = torch.nn.Softmax(dim=-1)

    def forward(self, input:torch.tensor) -> torch.tensor:
        lin = self.linear(input)
        out = self.output_layer(lin)
        return out 


class ModBertAttention(torch.nn.Module):
    '''
    Custom model using custom BERT-like layer
    '''
    def __init__(self, input_features, 
                 out_features, 
                 return_attention=False):

        super(ModBertAttention, self).__init__()
        self.attention = return_attention
        self.rnn = BertAttention(input_features)
        self.linear = torch.nn.Linear(input_features, out_features)
        self.output_layer = torch.nn.Softmax(dim=-1)

    def forward(self, input:torch.tensor) -> Union[torch.tensor, 
                                                   tuple[torch.tensor, 
                                                         torch.tensor]]:
        res, att = self.rnn(input)
        lin = self.linear(res)
        out = self.output_layer(lin)
        if self.attention:
            return att, out
        else:
            return out 


class ModSelfAttention(torch.nn.Module):
    '''
    Custom model using custom BERR-like layer
    '''
    def __init__(self, input_features, 
                 out_features,
                 return_attention=False):

        super(ModSelfAttention, self).__init__()
        self.attention = return_attention
        self.rnn = SelfAttention(input_features)
        self.linear = torch.nn.Linear(input_features, out_features)
        self.output_layer = torch.nn.Softmax(dim=-1)

    def forward(self, input:torch.tensor) -> Union[torch.tensor, 
                                                   tuple[torch.tensor, 
                                                         torch.tensor]]:
        res, att = self.rnn(input)
        lin = self.linear(res)
        out = self.output_layer(lin)
        if self.attention:
            return att, out
        else:
            return out 