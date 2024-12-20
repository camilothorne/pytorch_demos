from architectures.train_and_test import *
from architectures.bow_enc import *
from architectures.layers import ModBertAttention

import torch, scipy


def classif_exp(model:torch.nn.Module,
                train_data: np.array,
                val_data:np.array,
                test_data:np.array,
                labdict:dict,
                name:str,
                epochs:int) -> None:
   '''
   Train and test document classifier 
   '''

   print("--------------------")
   print(f"Classifier [{name}]")

   val_history = []
   loss_history =[]

   # Train data
   if scipy.sparse.issparse(train_data[0]):
      trainn_data = train_data[0].todense()
   else:
      trainn_data = train_data[0]

   # Val data
   if scipy.sparse.issparse(val_data[0]):
      vall_data = val_data[0].todense()
   else:
      vall_data = val_data[0]
   
   # Train labels
   if scipy.sparse.issparse(train_data[1]):
      trainn_labels = train_data[1].todense()
   else:
      trainn_labels = train_data[1]
   
   # Val labels
   if scipy.sparse.issparse(val_data[1]):
      vall_labels = val_data[1].todense()
   else:
      vall_labels = val_data[1]

   # Test data
   if scipy.sparse.issparse(test_data[0]):
      testt_data = test_data[0].todense()
   else:
      testt_data = test_data[0]
   
   # Test labels
   if scipy.sparse.issparse(test_data[1]):
      testt_labels = test_data[1].todense()
   else:
      testt_labels = test_data[1]

   # Use GPU for acceleration if avilable
   if torch.backends.mps.is_available():
      device_name = 'mps'
      '''
      The following instruction may freeze the OS:
      available memory on silicon Macs is usually ~ N - 12 GBs, where
      N is the maximum available RAM of an M1-M3 processor
      (e.g. if N = 24, then 12 GBs)
      '''
      torch.mps.set_per_process_memory_fraction(0.0)
   elif torch.cuda.is_available():
      device_name = 'cuda'
   else:
      device_name = 'cpu'

   print("--------------------")
   print(f"Using device (train and test): {device_name}")

   device = torch.device(device_name)
   model.to(memory_format=torch.channels_last)
   model.to(device)

   _, trained_model = train_variant(model, 
                        X_train=torch.tensor(trainn_data, dtype=torch.float32).to(device),
                        X_val=torch.tensor(vall_data, dtype=torch.float32).to(device),
                        y_train=torch.tensor(trainn_labels, dtype=torch.float32).to(device),
                        y_val=torch.tensor(vall_labels, dtype=torch.float32).to(device),
                        val_history=val_history,
                        loss_history=loss_history,
                        data_size=train_data[0].shape[0],
                        #batch_size=32, # standard batch size of 32
                        batch_size=16, # to avoid OOM errors on M2 chips
                        clip_value=100,
                        my_lr=0.0001,
                        my_momentum=0.009,
                        my_weight_decay=0.001,
                        my_loss=CustomCE(), # we use CE loss
                        val_size=val_data[0].shape[0],
                        loss_fun="CE loss", # name of loss function
                        epochs=epochs,
                 )

   plot_training_curve(loss_history=loss_history, 
                       val_history=val_history, 
                       path=f"./plots_and_stats/ce_loss_{name}.png", loss_fun="CE loss")

   print(f"Performance on test set")
   print("--------------------")
   print(test_variant(dnn=trained_model, 
               X_test=torch.tensor(testt_data, dtype=torch.float32).to(device), 
               y_test=np.asarray(testt_labels),
               labdict=labdict,
               path=f"./plots_and_stats/preds_{name}.csv",
               path_stats=f"./plots_and_stats/scores_{name}.csv",
               my_device_name=device_name
               )
         )


def bow_attention(epochs:int=30)->None:

   '''
   Pre-process data using BOW encoder
   '''

   print("--------------------")
   print(f"Dataset (BOW encoding)")
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

   model_1 = ModBertAttention(train_data[0].shape[1], train_data[1].shape[1], return_state=True)
   classif_exp(model_1, 
               train_data, 
               val_data, 
               test_data, 
               labeldict, 
               name="bow_bert_attention", 
               epochs=epochs)