import model
import numpy as np
from torch.autograd import Variable
import torch

def main():

    data=torch.randn(4,109,64) #batch,channels,input for each channel
    data=Variable(data)
    model_final = model.CNN()
    print(model_final)
    output = model_final(data)
    return output

if __name__ == '__main__':
    out=main()
    print(out.size())

