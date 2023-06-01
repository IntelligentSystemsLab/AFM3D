import numpy as np
import torch
import pandas as pd
from learner_AFM3DAver_SFD_densenet121 import Metanet
import time
import os
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
start = time.time() 

def main():
    epoch = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

    meta_net = Metanet(device)
    for i in range(epoch):
        if (i+1)%20 == 0 or (i+1)<=5:
            print("{} round training.".format(i+1))
        meta_net.meta_training(i+1)
        #if (i+1)%5 == 0:
        meta_net.Testing(i+1) 
   
if __name__ == '__main__':
    main()
    end = time.time() 
    run_time = end - start
    print('running time:',run_time,'s')