from train_rnn import trainRNN
#from train_mul import trainMul
import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required = True, help='Choose either "MUL" or"RNN"')
    parser.add_argument('--prop', type=str, required = True, help='ip or ea')
    parser.add_argument('--data', type=str, required = True, help='data directory')
    parsed_args = parser.parse_args()
    
    if parsed_args.model == 'RNN':
        trainRNN(parsed_args)
    elif parsed_args == 'MUL':
        print("Please use RNN only for now")#trainMul(parsed_args)