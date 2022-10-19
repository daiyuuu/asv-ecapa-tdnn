'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time
from tools import *
from ECAPAModel import ECAPAModel
import numpy as np

parser = argparse.ArgumentParser(description = "ECAPA_trainer")

## Training Model Settings
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## evaluation path/lists, save path
parser.add_argument('--eval_list',  type=str,   default="/media/dell/203A28373A280C7A/xlt/SASVC2022_Baseline-main/ECAPATDNN/data/val_list/veri_test.txt",  help='The path of the evaluation list')
parser.add_argument('--eval_path',  type=str,   default="/media/dell/203A28373A280C7A/xlt/voxceleb/wav",   help='The path of the evaluation data')
parser.add_argument('--score_save_path',  type=str,   default="/media/dell/203A28373A280C7A/xlt/ecapa-tdnn/exps/fbank/result/Vox1_O_N.txt",    help='Path to save the score.txt')
parser.add_argument('--initial_model',  type=str,   default="/media/dell/203A28373A280C7A/xlt/ecapa-tdnn/exps/fbank/model/model_0070.model",   help='Path of the infer_model')

## Model and Loss settings
parser.add_argument('--C',       type=int,   default=512,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=5994,   help='Number of speakers')


## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()

## Only do evaluation, the initial_model is necessary
if __name__ == "__main__":

    s = ECAPAModel(**vars(args))
    print("Model %s loaded from previous state!"%args.initial_model)
    s.load_parameters(args.initial_model)
    EER, minDCF = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
    print("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
    score_file = open(args.score_save_path,  "a+")
    score_file.write("%s model, EER %2.2f%%, minDCF %.4f%%\n"%(args.initial_model,EER, minDCF))
