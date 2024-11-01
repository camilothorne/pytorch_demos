from experiments.train_and_test import *
from experiments.bow_enc import *
from experiments.layers import ModBertAttention, ModSelfAttention
import torch


# def self_attention():
#    model = ModSelfAttention()
#    pass

def bow_classif_exp(model:torch.nn.Module, name:str) -> None:
   '''
   Train and test BOW document classifier 
   '''

   print("--------------------")
   print(f"BOW Classifier [{name}]")

   bow_encode = BoWEconde(path="./data/ecommerceDataset", colnames=['Label', 'Document'])
   print(bow_encode.get_raw_data.head())
   
   labeldict = bow_encode.get_label_dict()
   print(labeldict)
   
   train_data, test_data, val_data = bow_encode.split_data()

   val_history = []
   loss_history =[]

   _, trained_model = train_variant(model, 
                        X_train=train_data[0],
                        X_val=val_data[0],
                        y_train=train_data[1],
                        y_val=val_data[1],
                        val_history=val_history,
                        loss_history=loss_history,
                        data_size=train_data[0].shape[0],
                        batch_size=32, # standard batch size of 32
                        clip_value=100,
                        my_lr=0.0001,
                        my_momentum=0.009,
                        my_loss=torch.nn.CrossEntryLoss(), # we use CE loss
                        val_size=val_data[0],
                        loss_fun="CE loss", # name of loss function
                        epochs=30
                 )

   plot_training_curve(loss_history=loss_history, 
                       val_history=val_history, 
                       path=f"./plots_and_stats/ce_loss_bow_{name}.png", loss_fun="CE loss")

   print(test_variant(trained_model, test_data[0], test_data[1]))


if __name__ == '__main__':

   model_1 = ModBertAttention()
   model_2 = ModSelfAttention()

   bow_classif_exp(model_1, "bert_attention")
   bow_classif_exp(model_2, "self_attention")

