import torch
import argparse
from utils.model import APSClassifier

# Setting up the device for GPU usage
# Defining some key variables that will be used later on in the training
MAX_LEN = 512

from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'
print(f"DEVICE: {device}")


def parser_args():
    parser = argparse.ArgumentParser(description="model_path")
    parser.add_argument("--model_path", type=str,default='jinhwak/ASP_pseudo')
    parser.add_argument("--use_auth_token",type=str)
    args = parser.parse_args()
    return args

def model_loader(model_path,use_auth_token):
    model = APSClassifier(model_path,use_auth_token= use_auth_token)
    return model


if __name__ == "__main__":
    args = parser_args()
    model = model_loader(args.model_path,args.use_auth_token)
    while True:
        dialogue = input("Enter Text or Dialogue ")
        if dialogue == 'exit':
            break
        dialogue = "Human: " + dialogue
        print(f"Label : {model.predict(dialogue)}")

