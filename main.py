from experiments.train_and_test import *
from experiments.bow_enc import *
from experiments.layers import ModBertAttention, ModSelfAttention
import torch


# def self_attention():
#    model = ModSelfAttention()
#    pass

def classif_exp(model:torch.nn.Module,
                train_data: np.array,
                val_data:np.array,
                test_data:np.array,
                labdict:dict,
                name:str) -> None:
   '''
   Train and test BOW document classifier 
   '''

   print("--------------------")
   print(f"Classifier [{name}]")

   val_history = []
   loss_history =[]

   _, trained_model = train_variant(model, 
                        X_train=torch.tensor(train_data[0].todense(), dtype=torch.float32),
                        X_val=torch.tensor(val_data[0].todense(), dtype=torch.float32),
                        y_train=torch.tensor(train_data[1].todense(), dtype=torch.float32),
                        y_val=torch.tensor(val_data[1].todense(), dtype=torch.float32),
                        val_history=val_history,
                        loss_history=loss_history,
                        data_size=train_data[0].shape[0],
                        batch_size=32, # standard batch size of 32
                        clip_value=100,
                        my_lr=0.0001,
                        my_momentum=0.009,
                        my_loss=torch.nn.CrossEntropyLoss(), # we use CE loss
                        val_size=val_data[0].shape[0],
                        loss_fun="CE loss", # name of loss function
                        epochs=30
                 )

   plot_training_curve(loss_history=loss_history, 
                       val_history=val_history, 
                       path=f"./plots_and_stats/ce_loss_bow_{name}.png", loss_fun="CE loss")

   print(f"Performance on test set")
   print("--------------------")
   print(test_variant(dnn=trained_model, 
                      X_test=torch.tensor(test_data[0].todense(), dtype=torch.float32), 
                      y_test=np.asarray(test_data[1].todense()),
                      labdict=labdict,
                      path=f"./plots_and_stats/preds_{name}.csv"))


if __name__ == '__main__':

   '''
   Pre-process data
   '''

   print("--------------------")
   print(f"Dataset")
   print("--------------------")
   bow_encode = BoWEconde(path="./data/ecommerceDataset.csv", 
                          sep=',')

   print(bow_encode.get_raw_data().head())
   print("--------------------")
   labeldict = bow_encode.get_label_dict()
   print('Labels:', labeldict)
   train_data, test_data, val_data = bow_encode.split_data()
   print('Dimensions of training data: (data and labels):', train_data[0].shape, train_data[1].shape)

   '''
   Run experiments
   '''

   model_1 = ModBertAttention(train_data[0].shape[1], train_data[1].shape[1])
   classif_exp(model_1, train_data, val_data, test_data, labeldict, "bow_bert_attention")
   
   #model_2 = ModSelfAttention(train_data[0].shape[1], train_data[1].shape[1])
   #classif_exp(model_2, train_data, val_data, test_data, "bow_self_attention")

