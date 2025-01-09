from experiments import bow_att, bow_linear, one_hot_att, one_hot_cnn
import argparse
import wandb

'''
Argument parser
'''

parser = argparse.ArgumentParser(
                    prog='python main.py',
                    description='CLI for running the PyTorch experiments.',
                    epilog='For more information, refer to the README.')
parser.add_argument('-e', '--exp', required=True,type=str,
                    help='choose an experiment: [bow_base, one_hot_base, bow_att, one_hot_att]')
parser.add_argument('-i', '--iter', required=True, type=int,
                    help='number of epochs (integer)')
parser.add_argument('-s', '--sc', required=False, type=str,
                    help='if set to yes: returns attention scores for [bow_base, bow_att, one_hot_att]')
parser.add_argument('-w', '--wb', required=False, type=str,
                    help='if set to yes: saves plots and checkpoints on weights & biases')

'''
Parse arguments and options
'''

args   = parser.parse_args()

# Recover number of epochs
epochs = args.iter

# If scores are chosen, the model will return
# attention or feature scores
if args.sc == 'yes':
    scores = True
else:
    scores = False

# If wandb is chosen, the experiment (performance metrics, plots.
# checkpoints will be logged on wandb)
if args.wb == 'yes':
    my_wandb = True
    # Login to wabdb
    wandb.login() 
    run = wandb.init(
      # Set the project where this run will be logged
      project="pytorch-demos",
      # Track epochs, names and kind of experiment
      config={
        "epochs": epochs,
        "experiment": args.exp,
        "scores_returned": scores},
      )
else:
    my_wandb = False

# Recover experiment
if args.exp == 'bow_base':
    bow_linear.bow_linear(epochs, scores, my_wandb)
elif args.exp == 'one_hot_base':
    one_hot_cnn.one_hot_cnn(epochs, my_wandb)
elif args.exp == 'bow_att':
    bow_att.bow_attention(epochs, scores, my_wandb)
elif args.exp == 'one_hot_att':
    one_hot_att.one_hot_att(epochs, scores, my_wandb)
else:
    print("Invalid. Please check the help.")

# Save weights & biases if wandb is chosen and experiment is complete
if my_wandb:
   if args.exp == 'bow_base':
      wandb.save('models/*bow*log*')
      wandb.save('plots_and_stats/*bow*log*')
   elif args.exp == 'one_hot_base':
      wandb.save('models/*onehot*conv*')
      wandb.save('plots_and_stats/*onehot*conv*')
   elif args.exp == 'bow_att':
      wandb.save('models/*bow*att*')
      wandb.save('plots_and_stats/*bow*att*')
   elif args.exp == 'one_hot_att':
      wandb.save('models/*onehot*att*')
      wandb.save('plots_and_stats/*onehot*att*')
   run.finish()
else:
    pass