import math, copy, os.path
import torch
import numpy as np
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
from torch.nn.modules.module import _addindent
from sklearn.metrics import classification_report


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
    plt.ylabel(loss_fun + " loss")
    plt.xlabel("Validation batch")
    plt.xticks(range(0,len(val_history)))
    plt.title(f"Training loss across epochs")
    plt.legend(loc='upper right')
    plt.savefig(path)


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
                  my_loss:torch.nn.Functional,
                  val_size:int,
                  loss_fun:str, # name of loss function
                  epochs:int
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
    optimizer   = torch.optim.SGD(dnn.parameters(), lr=my_lr, momentum=my_momentum)

    # Hold the best model
    best_loss        = np.inf   # init to infinity
    best_weights     = None

    # Detect anomalies
    torch.autograd.set_detect_anomaly(True) # check for anomaly in gradients

    print("--------------------")
    print(f"SGD for {epochs} epochs, with batch size {batch_size}:")
    for i in tqdm(range(epochs)):
        
        # Train
        dnn.train()
        start = 0
        count = 0

        while data_size - start > batch_size:
            optimizer.zero_grad()
            out = dnn(X_train[start:start+batch_size]) # forward pass
            loss = my_loss(out, y_train[start:start+batch_size])
            loss.backward()
            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(dnn.parameters(), clip_value)
            optimizer.step()
            start = start + batch_size
            count = count + 1

        # Validate
        dnn.eval()
        estart = 0
        ecount = 0
        while val_size - estart > batch_size:
            y_pred = dnn(X_val[estart:estart+batch_size])
            v_loss = my_loss(y_pred, y_val[estart:estart+batch_size])
            if ((i%10 == 0) & (ecount%20 == 0)):
                loss_history.append(loss.item())
                print(" - loss at epoch {} and test batch {} is: {:.10f}".format(i, ecount, v_loss.item()))
                val_history.append(v_loss.item())
                if v_loss.item() < best_mse:
                    best_mse = v_loss.item()
                    best_weights = copy.deepcopy(dnn.state_dict())
            estart = estart + batch_size
            ecount = ecount + 1

    print("--------------------")
    print(loss_fun + "(best): %.2f" % best_loss)
    print(loss_fun + "(best): %.2f" % np.sqrt(best_loss))
    print("--------------------")

    return best_weights, dnn


def test_variant(dnn:torch.nn.Module, 
                 X_test:np.Array, y_test:np.Array) -> str:
    '''
    Measure classification performance
    '''
    
    y_pred = dnn(X_test)
    return classification_report(y_pred, y_test)