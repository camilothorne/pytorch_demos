from experiments import bow_att, bow_linear, one_hot_att, one_hot_cnn
import argparse


parser = argparse.ArgumentParser(
                    prog='python main.py',
                    description='A script for running PyTorch experiments.',
                    epilog='For more information, refer to README.')
parser.add_argument('-e', '--exp', required=True,
                    help='choose an experiment: [bow_base, one_hot_base, bow_att, one_hot_att]')
args = parser.parse_args()


if args.exp == 'bow_base':
    bow_linear.bow_linear()
elif args.exp == 'one_hot_base':
    one_hot_cnn.one_hot_cnn()
elif args.exp == 'bow_att':
    bow_att.bow_attention()
elif args.exp == 'one_hot_att':
    one_hot_att.one_hot_att()
else:
    print("Invalid. Please choose one of: [bow_base, one_hot_base, bow_att, one_hot_att].")