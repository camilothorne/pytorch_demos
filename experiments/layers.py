import math
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
        #self.reset_parameters()

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

        E_t = H_t * H_t^T
        Alpha_t = softmax(E_t)
        C = \sum_t Alpha_t * H_t
    
    '''
    def __init__(self, input_dim:int) -> None:
        '''
        Constructor
        '''
        super(SelfAttention,self).__init__()
        self.hidden = torch.nn.Linear(input_dim, input_dim)
        #self.reset_parameters()

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
        c = self.hidden(input)
        e = torch.matmul(input.transpose(0,1), c)

        # Compute the weights
        alpha = torch.nn.functional.softmax(e, dim=1)
        
        # Compute the context vector
        context = torch.matmul(input, alpha)
        output = input * torch.sum(context, dim=1)[:,None]

        return output, context


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
        
    
'''
Convolutions for one-hot matrixes
'''


class TextConv1D(torch.nn.Module):
    '''
    Custom ConvNet for text classification of sentence one-hot encodings

        Conv1d(in_channels, out_channels, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, 
                bias=True, padding_mode='zeros', device=None, 
                dtype=None)

    '''
    def __init__(self, input_features, out_features):
        '''
        This model takes in two basic parameters

        - input_features:   the word embedding dimension
        - out_features:     the number of classes 

        '''
        super(TextConv1D, self).__init__()
        self.layer1 = torch.nn.Sequential(
                torch.nn.Conv1d(input_features[1], out_channels=10, kernel_size=3),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(10))
        self.layer2 = torch.nn.Flatten()
        self.layer3 = torch.nn.Sequential(
                torch.nn.Linear(int(math.floor(input_features[2]/10)*10),100),
                torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(
                torch.nn.Linear(100,out_features),
                torch.nn.Softmax(dim=-1))

    def forward(self, x):

        #print('input', x.shape)
        out = self.layer1(x)
        #print('cnn layer', out.shape)
        out = self.layer2(out)
        #print('flatten layer', out.shape)
        out = self.layer3(out)
        #print('linear layer', out.shape)
        out = self.layer4(out)
        #print('softmax layer', out.shape)
        return out