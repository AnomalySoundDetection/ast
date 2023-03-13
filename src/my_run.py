import os 
import torch
from models import ASTModel 
from torchsummary import summary

# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'  
# assume each input spectrogram has 100 time frames
input_tdim = 200
# assume the task has 527 classes
label_dim = 527
# create a pseudo input: a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins 
test_input = torch.rand([10, input_tdim, 128]) 
# create an AST model
ast_mdl = ASTModel(input_tdim=input_tdim, audioset_pretrain=True)
#summary(ast_mdl, (100, 128))

#print(ast_mdl)
ast_mdl = ast_mdl.to(torch.device('cuda:0'))
#test_output = ast_mdl(test_input) 
# output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes. 
#print(test_output.shape) 