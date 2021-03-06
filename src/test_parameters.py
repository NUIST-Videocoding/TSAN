import torch
import os
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


global model
model = model.Model(args, checkpoint)
blank = ' '
print('-' * 90)
print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
      + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
      + ' ' * 3 + 'number' + ' ' * 3 + '|')
print('-' * 90)
num_para = 0
type_size = 4  ##如果是浮点数就是4

for index, (key, w_variable) in enumerate(model.named_parameters()):
    if len(key) <= 30:
        key = key + (30 - len(key)) * blank
    shape = str(w_variable.shape)
    if len(shape) <= 40:
        shape = shape + (40 - len(shape)) * blank
    each_para = 1
    for k in w_variable.shape:
        each_para *= k
    num_para += each_para
    str_num = str(each_para)
    if len(str_num) <= 10:
        str_num = str_num + (10 - len(str_num)) * blank

    print('| {} | {} | {} |'.format(key, shape, str_num))
print('-' * 90)
print('The total number of parameters: ' + str(num_para))
print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
print('-' * 90)


