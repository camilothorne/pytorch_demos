import math, copy, os.path
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
from torch.nn.modules.module import _addindent
from sklearn.metrics import classification_report

import wandb


class CustomCE(torch.nn.Module):
    '''
    Custom CE loss that operates on:

    - softmax layers
    - one hot encodings of labels

        CE = - 1/n [ \sum_i^n ( X_i * ln(P_i) ) ]

    where X_i and P_i are 4-dimensional vectors (true labels and their 
    probabilities) 

    (Note: PyTorch CrossEntropyLoss operates on logits and single labels, 
    somehow!)

    '''
    
    def __init__(self):
        super(CustomCE, self).__init__()

    def forward(self, inputs, targets):
        loss = -torch.sum (targets * torch.log(inputs))
        return loss.mean()


def torch_summarize(model:torch.nn.Module,
                    show_weights=True,
                    show_parameters=True) -> str:
    """
    Summarizes torch model by showing trainable parameters and weights
    """
    
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)
        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])
        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'
    tmpstr = tmpstr + ')'
    return tmpstr


def write_model_to_file(model:torch.nn.Module, 
                        path:str) -> None:
    '''
    Save full model
    '''
    
    torch.save(model, path)


def read_model_from_file(path:str) -> torch.nn.Module:
    '''
    Load model - once returned, call 
        
        model.eval()

    '''
    
    model = torch.load(path, weights_only=False)
    return model


def write_model_state_to_file(state_dict:dict, 
                              path:str) -> None:
    '''
    Save model state only
    '''
    
    torch.save(state_dict, path)


def read_model_state_from_file(model:torch.nn.Module, 
                          state_dict:dict, 
                          path:str) -> torch.nn.Module:
    '''
    Load model from model state -
    An empy model graph of the right kind needs to
    be instantiated fisrt via
        
        model = TheModelClass(*args, **kwargs)

    Once returned, call 
    
        model.eval()

    '''

    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def plot_training_curve(loss_history:list,
                val_history:list,
                path:str,
                loss_fun:str) -> None:
    '''
    Display learning curve
    '''
    
    plt.plot(loss_history, label="train")
    plt.plot(val_history, label="val")
    plt.ylabel(loss_fun)
    plt.xlabel("Validation batch")
    plt.xticks(range(0,len(val_history)), rotation=45, fontsize=8)
    plt.margins(0.2)
    plt.title(f"Training loss")
    plt.legend(loc='upper right')
    plt.savefig(path)
    plt.close()


def train_variant(dnn:torch.nn.Module,
                  X_train:np.array,
                  X_val:np.array,
                  y_train:np.array,
                  y_val:np.array,
                  val_history:list,
                  loss_history:list,
                  data_size:int,
                  batch_size:int,
                  clip_value:int,
                  my_lr:float,
                  my_momentum:float,
                  my_weight_decay:float, # regularization for L2 loss
                  my_loss:torch.nn.Module,
                  val_size:int,
                  loss_fun:str, # name of loss function
                  epochs:int,
                  scores:bool=False,
                  my_wandb:bool=False,
    ) -> tuple[dict, torch.nn.Module]:
    '''
    Train and validate model -
    this function loads model weights and input 
    tensors into CPU
    '''

    print("--------------------")
    print("Model and parameters:")
    print(torch_summarize(dnn))

    # Choose optimizer and loss function (MSE)
    optimizer   = torch.optim.SGD(dnn.parameters(), 
                                  lr=my_lr, 
                                  momentum=my_momentum, 
                                  weight_decay=my_weight_decay)

    # Hold the best model
    best_acc         = 0
    best_loss        = np.inf   # init to infinity
    best_weights     = None

    # Detect anomalies
    torch.autograd.set_detect_anomaly(True) # check for anomaly in gradients

    if my_wandb:
        wandb.log(
                {
                    "batch_size": batch_size,
                    "train_datasize": data_size,
                    "val_datasize": val_size
                }
        )

    print("--------------------")
    print(f'[Return scores/attention? {scores}]')
    print("--------------------")
    print(f'Learning rate: {my_lr}')
    print(f'Momentum: {my_momentum}')
    print(f'Weight decay: {my_weight_decay}')
    print(f'Clip value: {clip_value}')
    print(f'Training examples: {data_size}')
    print(f'Validation examples: {val_size}')
    print("--------------------")
    print(f"SGD for {epochs} epochs, with batch size {batch_size}:")

    for i in tqdm(range(epochs)):
        
        # 1) Train
        dnn.train()
        start = 0
        count = 0

        while data_size - start > batch_size:

            optimizer.zero_grad()
            if scores:
                   #print(scores)
                   y_pred = dnn(X_train[start:start+batch_size])[-1] # forward pass
            else:
                   y_pred = dnn(X_train[start:start+batch_size]) # forward pass

            #print(y_pred.shape)
            #print(y_train[start:start+batch_size].shape)

            loss   = my_loss(y_pred, y_train[start:start+batch_size]) # training loss
            loss.backward()
            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(dnn.parameters(), clip_value)
            optimizer.step()
            start = start + batch_size
            count = count + 1

        # 2) Validate
        dnn.eval()
        estart = 0
        ecount = 0

        while ((val_size - estart) > batch_size):

            if scores:
                   y_pred = dnn(X_val[estart:estart+batch_size])[-1] # eval
            else:
                   y_pred = dnn(X_val[estart:estart+batch_size]) # eval
            v_loss = my_loss(y_pred, y_val[estart:estart+batch_size]) # validation loss
            v_acc = (torch.argmax(y_pred, 1) == torch.argmax(y_val[estart:estart+batch_size], 
                                                           1)).float().mean() # validation accuracy
            
            if ((i%10 == 0) & (ecount%20 == 0)):
                
                # We collect the stats every now and then
                loss_history.append(loss.item())
                print(" - val. loss at epoch {} and batch {} is: {:.10f}".format(i, ecount, v_loss.item()))
                print(" - val. accuracy at epoch {} and batch {} is: {:.10f}".format(i, ecount, v_acc))
                val_history.append(v_loss.item())
                if v_loss.item() < best_loss:
                    best_loss = v_loss.item()
                    best_weights = copy.deepcopy(dnn.state_dict())
                if v_acc > best_acc:
                    best_acc = v_acc
                # Uncomment for debugging
                #print(f' ** val. labels: {y_val[estart:estart+batch_size]}')
                #print(f' ** predicted prob. distrib.: {y_pred}')
            
            estart = estart + batch_size
            ecount = ecount + 1

    print("--------------------")
    print("Accuracy (best): %.2f" % best_acc)
    print(loss_fun + " (best): %.2f" % best_loss)
    print("--------------------")

    if my_wandb:
        wandb.log(
                {
                    "epoch": i,
                    "train_loss": loss.item(),
                    "val_acc": v_acc,
                    "val_acc_best": best_acc,
                    "val_loss": v_loss.item(),
                    "lr":my_lr,
                    "momentum": my_momentum,
                    "weight_decay": my_weight_decay,
                    "val_loss_best": best_loss
                }
        )

    return best_weights, dnn


def test_variant(dnn:torch.nn.Module, 
                 X_test:np.array, 
                 y_test:np.array,
                 labdict:dict,
                 path:str,
                 path_stats:str,
                 my_device_name:str,
                 y_index:np.array,
                 y_df:pd.DataFrame,
                 ) -> str:
    '''
    Measure classification performance (basic)
    '''

    # Label dictionary
    labd = {v:k for k,v in labdict.items()}
    
    # Predictions
    dnn.eval() # call model in eval mode

    # Get predictions
    y_pred = dnn(X_test)
    
    # Tranfer tensors to CPU (if in GPU) else do nothing
    if my_device_name != 'cpu':
        y_pred = torch.argmax(y_pred, 1).cpu()
    else:
        y_pred = torch.argmax(y_pred, 1)
    
    # For debugging
    #print(f'Predictions: {y_pred.shape}')
    #print(f'Gold: {y_test.shape}')

    # Gold
    y_test = np.argmax(y_test, axis=1)

    # Serialize predictions and gold labels
    y_df = y_df[['Document']]
    y_df['Index'] = y_df.index
    df = pd.DataFrame({"pred": y_pred.detach().numpy(), 
                      "gold": y_test})
    df['Gold'] = df.gold.astype(int).map(labd)
    df['Pred'] = df.pred.astype(int).map(labd)
    df['Index'] = y_index
    df = df.merge(y_df, on='Index', how='inner')[['Gold', 'Pred', 'Index', 'Document']]

    print('Predictions:\n')
    print(df.head(3))

    # Save predictions to CSV file
    df.to_csv(path, index=False)
    
    # Print and serialize the classification report
    result = classification_report(df.Gold, 
                                   df.Pred, 
                                   zero_division=0)
    result_dict = classification_report(df.Gold, 
                                   df.Pred, 
                                   zero_division=0,
                                   output_dict=True)
    df_res = df = pd.DataFrame(result_dict).transpose()
    df_res.to_csv(path_stats)

    return result


def test_variant_one_hot(dnn:torch.nn.Module, 
                 X_test:np.array, 
                 y_test:np.array,
                 labdict:dict,
                 path:str,
                 path_stats:str,
                 my_device_name:str,
                 y_index:np.array,
                 y_df:pd.DataFrame,
                 ) -> str:
    '''
    Measure classification performance
    with one hot encodings and attention.
    '''

    # Label dictionary
    labd = {v:k for k,v in labdict.items()}
    
    # Predictions
    dnn.eval() # call model in eval mode

    # Get predictions w. attention scores
    y_att, y_pred = dnn(X_test)
    
    # Tranfer tensors to CPU (if in GPU) else do nothing
    if my_device_name != 'cpu':
        y_pred = torch.argmax(y_pred, 1).cpu()
    else:
        y_pred = torch.argmax(y_pred, 1)

    # For debugging
    #print(f'Attention: {y_att.shape}')
    #print(f'Predictions: {y_pred.shape}')
    #print(f'Gold: {y_test.shape}')

    # Normalize attention across sequence length
    # using softmax
    # y_att = torch.softmax(y_att, 1)

    # Gold
    y_test = np.argmax(y_test, axis=1)

    # Serialize predictions and gold labels
    y_df = y_df[['Document']]
    y_df['Index'] = y_df.index
    df = pd.DataFrame({"pred": y_pred.detach().numpy(), 
                      "gold": y_test})
    df['Gold'] = df.gold.astype(int).map(labd)
    df['Pred'] = df.pred.astype(int).map(labd)
    df['Index'] = y_index
    df = df.merge(y_df, on='Index', how='inner')[['Gold', 'Pred', 'Index', 'Document']]

    # Save attention scores
    if my_device_name == 'cpu':
        df['Scores'] = list(y_att.detach().numpy())
    else:
        df['Scores'] = list(y_att.detach().cpu().numpy())

    # Get truncated test sequences per example
    # (returns a list of tokens, one per score)
    df['Seq'] = df['Document'].apply(lambda x: x.lower().split(" ")[0:y_att.shape[1]])

    print('Predictions:\n')
    print(df.head(3))

    # Save predictions to CSV file
    df.to_csv(path, index=False)
    
    # Print and serialize the classification report
    result = classification_report(df.Gold, 
                                   df.Pred, 
                                   zero_division=0)
    result_dict = classification_report(df.Gold, 
                                   df.Pred, 
                                   zero_division=0,
                                   output_dict=True)
    df_res = df = pd.DataFrame(result_dict).transpose()
    df_res.to_csv(path_stats)

    return result


def test_variant_scores(dnn:torch.nn.Module, 
                 X_test:np.array, 
                 y_test:np.array,
                 labdict:dict,
                 path:str,
                 path_stats:str,
                 my_device_name:str,
                 y_index:np.array,
                 features:list,
                 y_df:pd.DataFrame,
                 ) -> str:
    '''
    Measure classification performance with
    BOW features and feature importance scores.
    '''

    # Label dictionary
    labd = {v:k for k,v in labdict.items()}
    
    # Predictions
    dnn.eval() # call model in eval mode

    # Get predictions w. (attention) scores
    y_sc, y_pred = dnn(X_test)
    
    # Tranfer tensors to CPU (if in GPU) else do nothing
    if my_device_name != 'cpu':
        y_pred = torch.argmax(y_pred, 1).cpu()
    else:
        y_pred = torch.argmax(y_pred, 1)
    
    # For debugging
    #print(f'Attention: {y_sc.shape}')
    #print(f'Predictions: {y_pred.shape}')
    #print(f'Gold: {y_test.shape}')

    # Gold
    y_test = np.argmax(y_test, axis=1)

    # Serialize predictions and gold labels
    y_df = y_df[['Document']]
    y_df['Index'] = y_df.index
    df = pd.DataFrame({"pred": y_pred.detach().numpy(), 
                      "gold": y_test})
    df['Gold'] = df.gold.astype(int).map(labd)
    df['Pred'] = df.pred.astype(int).map(labd)
    df['Index'] = y_index
    df = df.merge(y_df, on='Index', how='inner')[['Gold', 'Pred', 'Index', 'Document']]

    # Save (attention) scores
    if my_device_name == 'cpu':
        df['Scores'] = list(y_sc.detach().numpy())
    else:
        df['Scores'] = list(y_sc.detach().cpu().numpy())

    # Get features per example
    # (returns a list of tokens, one per score)
    df['Feat'] = df.Index.apply(lambda x: str(x).replace(str(x), str(features)))

    print('Predictions:\n')
    print(df.head(3))

    # Save predictions to CSV file
    df.to_csv(path, index=False)
    
    # Print and serialize the classification report
    result = classification_report(df.Gold, 
                                   df.Pred, 
                                   zero_division=0)
    result_dict = classification_report(df.Gold, 
                                   df.Pred, 
                                   zero_division=0,
                                   output_dict=True)
    df_res = df = pd.DataFrame(result_dict).transpose()
    df_res.to_csv(path_stats)

    return result