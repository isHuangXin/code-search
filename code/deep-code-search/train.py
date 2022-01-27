import os
import sys
import argparse



def parser_args():
    parser = argparse.ArgumentParser("Train and Validate The Code Search (Embedding) Model")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--dataset', type='str', default='JointEmbeder', help='model name: JointEmbeder, SelfAttnModel')

if __name__ == '__main__':
    args = parser_args()
