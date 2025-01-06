from experiments import bow_att, bow_linear, one_hot_att, one_hot_cnn
import argparse


parser = argparse.ArgumentParser(
                    prog='python main.py',
                    description='A script for running PyTorch experiments.',
                    epilog='For more information, refer to README.')
parser.add_argument('-e', '--exp', required=True,type=str,
                    help='choose an experiment: [bow_base, one_hot_base, bow_att, one_hot_att]')
parser.add_argument('-i', '--iter', required=True, type=int,
                    help='number of epochs (integer)')
parser.add_argument('-s', '--sc', required=False, type=str,
                    help='if yes: returns attention scores for one_hot_att')
args = parser.parse_args()

epochs = args.iter
if args.sc == 'yes':
    scores = True
else:
    scores = False

if args.exp == 'bow_base':
    bow_linear.bow_linear(epochs, scores)
elif args.exp == 'one_hot_base':
    one_hot_cnn.one_hot_cnn(epochs)
elif args.exp == 'bow_att':
    bow_att.bow_attention(epochs, scores)
elif args.exp == 'one_hot_att':
    one_hot_att.one_hot_att(epochs, scores)
else:
    print("Invalid. Please check the help.")