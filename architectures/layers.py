import math
import torch
from typing import Union


'''
Custom attention layers
'''


class BertAttention(torch.nn.Module):
    '''
    The definition of transformer self-attention is as follows:

        out = softmax( (Q * K^T) / sqrt(d)) * V

    where Q, V and K are linear transforms of the input
    resp. activation of previous layer.

    Derived from 
    
       - https://arxiv.org/pdf/1706.03762 (paper)
       - https://stackoverflow.com/questions/76648620/how-do-i-implement-this-attention-layer-in-pytorch (blog)
    
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

        # For debugging
        # print(f'Attention weights: {attention_weights.shape}')
        # print(f'Attention output: {out.shape}')

        return out
    

# class SelfAttention(torch.nn.Module):
#     '''
#     Class implementing self-attention from 
#        
#         https://arxiv.org/pdf/1512.08756.pdf
#    
#     Adapted from attention for RNNs as described above.
#    
#     '''
#     def __init__(self, input_dim:int) -> None:
#         '''
#         Constructor
#         '''
#         super(SelfAttention,self).__init__()
#         self.hidden = torch.nn.Linear(input_dim, input_dim)
#         #self.reset_parameters()
#
#     def reset_parameters(self) -> None:
#         '''
#         Init parameters with random weights
#         '''
#         stdv = 1.0 / math.sqrt(self.feature_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, +stdv)      
# 
#     def forward(self, input:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         '''
#         Forward pass for back-propagation
#         '''
#         # Alignment scores
#         c = self.hidden(input)
#         e = torch.matmul(input.transpose(0,1), c)
#
#         # Compute the weights
#         alpha = torch.nn.functional.softmax(e, dim=1)
#       
#         # Compute the context vector
#         context = torch.matmul(input, alpha)
#         output = input * torch.sum(context, dim=1)[:,None]
#
#         return output, context
    

class SeqAttention(torch.nn.Module):
    '''
    Attention layer for sequential inputs (biLSTMs, RNNs, etc.), derived from

        - https://arxiv.org/pdf/1805.12307 (paper)
        - https://github.com/gentaiscool/lstm-attention (GitHub)

    Works in principle for RNN units H_t. The definitions are as follows:

        E_t     = a(H_t) \cdot a(H_t)^T
        Alpha_t = softmax(E_t)
        C       = \sum_t Alpha_t \cdot H_t  

    '''    
    def __init__(self, input_dim:int) -> None:
        '''
        Constructor
        '''
        super(SeqAttention, self).__init__()
        self.hidden = torch.nn.Linear(input_dim, input_dim)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        '''
        Forward pass
        '''
        # Linear transformation
        hid = self.hidden(input)

        # Alignment scores
        e = torch.matmul(hid.transpose(1,2), hid)

        # Compute the weights
        alpha = torch.nn.functional.softmax(e, dim=1)

        # Compute context
        context = torch.sum(
                        torch.matmul(alpha, input.transpose(1,2)), 
                        dim=1)

        # For debugging
        #print('H_t (query):', input.shape)
        #print('a(H_t) (value):', hid.shape)
        #print('E_t (key):', e.shape)
        #print('Alpha_t (key):', alpha.shape)
        #print('context C (result):', context.shape)
        
        return context


'''
Self attention models for BOE encoded data (with linear baseline)
'''


class ModLogReg(torch.nn.Module):
    '''
    Logistic regression in Pytorch (single layer followed by softmax)
    '''
    def __init__(self, input_features, 
                 out_features,
                 return_scores=False):

        super(ModLogReg, self).__init__()
        self.scores = return_scores
        self.linear = torch.nn.Linear(input_features, out_features)
        self.output_layer = torch.nn.Softmax(dim=-1)

    def forward(self, input:torch.tensor) -> torch.tensor:
        lin = self.linear(input)
        out = self.output_layer(lin)
        if self.scores:
            return input, out
        else:
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
        res = self.rnn(input)
        lin = self.linear(res)
        out = self.output_layer(lin)
        if self.attention:
            #att = torch.sum(att, dim=1)
            return res, out
        else:
            return out 


# class ModSelfAttention(torch.nn.Module):
#     '''
#     Custom model using custom BERT-like layer
#     '''
#     def __init__(self, input_features, 
#                  out_features,
#                  return_attention=False):
#
#         super(ModSelfAttention, self).__init__()
#         self.attention = return_attention
#         self.rnn = SelfAttention(input_features)
#         self.linear = torch.nn.Linear(input_features, out_features)
#         self.output_layer = torch.nn.Softmax(dim=-1)
#
#     def forward(self, input:torch.tensor) -> Union[torch.tensor, 
#                                                    tuple[torch.tensor, 
#                                                          torch.tensor]]:
#         res, att = self.rnn(input)
#         lin = self.linear(res)
#         out = self.output_layer(lin)
#         if self.attention:
#             return att, out
#         else:
#             return out 
        
    
'''
CNN classification models for one-hot encoded data
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

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


'''
Self attention classification model for one-hot encoded (sequential) data 
'''


# class TextBiLSTMAttention(torch.nn.Module):
#     '''
#     Custom biLSTM w. attention model for 
#     one-hot encoding / sequence data, based on this class

#         LSTM(input_size, hidden_size, num_layers=1, bias=True, 
#             batch_first=False, dropout=0.0, bidirectional=True, 
#             proj_size=0, device=None, dtype=None)
#
#     '''
#     def __init__(self, in_features, 
#                  latent_dim, out_features, 
#                  return_state=False):
#         '''
#         Key parameters
#         - in_features     (word_dim, length) tuple
#         - out_features:   ouput features
#         - latent_dim:     hidden state dimenstion
#         '''
#         super(TextBiLSTMAttention, self).__init__()
#         self.ret_state = return_state
#         self.rnn = torch.nn.LSTM(input_size=in_features[2], 
#                                      hidden_size=latent_dim,
#                                      bidirectional=True)
#         self.attention = SeqAttention(latent_dim * 2)
#         self.linear = torch.nn.Linear(in_features[1], out_features)
#         self.output_layer = torch.nn.Softmax(dim=-1)
#       
#     def forward(self, x):
#         lstm_out, (h_s, c_s)    = self.rnn(x) # (batch_size, embedding_dim, sequence_length)
#         #print('input', lstm_out.shape)
#         out                     = self.attention(lstm_out)
#         #print('context', out.shape)
#         out                     = self.linear(out)
#         #print('linear', out.shape)
#         out                     = self.output_layer(out)
#         #print('output', out.shape)
#         if self.ret_state:
#             return h_s, c_s, out
#         else:
#             return out
        
class TextLSTMAttention(torch.nn.Module):
    '''
    Custom LSTM w. attention model for 
    one-hot encoding / sequence data, based on this class

        LSTM(input_size, hidden_size, num_layers=1, bias=True, 
            batch_first=False, dropout=0.0, bidirectional=False, 
            proj_size=0, device=None, dtype=None)
    
    '''
    def __init__(self, in_features, 
                 latent_dim, out_features, 
                 return_state=False,
                 return_attention=False):
        '''
        Key parameters

        - in_features     (word_dim, length) tuple
        - out_features:   ouput features
        - latent_dim:     hidden state dimenstion

        '''
        super(TextLSTMAttention, self).__init__()
        self.ret_state = return_state
        self.ret_att = return_attention
        self.rnn = torch.nn.LSTM(input_size=in_features[1], 
                                     hidden_size=latent_dim,
                                     bidirectional=False)
        self.attention = SeqAttention(latent_dim)
        self.linear = torch.nn.Linear(in_features[2], out_features)
        self.output_layer = torch.nn.Softmax(dim=-1)
        
    def forward(self, x):
        '''
        Forward pass:

            - input shape: (B, V, T)
            - attention shape: (B, T)
            - output shape: (B, L)
        
        where B = batch size, embedding dimension, L = number of labels and
        L = sequence length    
        '''

        x                       = x.transpose(1,2) # (batch_size, sequence_length, embedding_dim)
        lstm_out, (h_s, c_s)    = self.rnn(x) # (batch_size, sequence_length, hidden_dim)     
        att_out                 = self.attention(lstm_out) # (batch_size, sequence_length)
        out                     = self.linear(att_out) # (batch_size, num_labels)
        out                     = self.output_layer(out) # (batch_size, num_labels)

        # For debugging
        #print('Input (transposed):', x.shape) 
        #print('LSTM layer:', lstm_out.shape)
        #print('Attention layer:', att_out.shape)
        #print('Output (softmax):', out.shape)

        if self.ret_state:
            return h_s, c_s, out
        elif self.ret_att:
            return att_out, out
        else:
            return out