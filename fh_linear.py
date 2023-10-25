import torch
import torch.nn as nn
from torch.autograd import Function
from pytorch_apis import Matrix_multiplication, Trans_pose
from gp_apis import gp_Matrix_multiplication, gp_Trans_pose
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class FhLinearFunction(Function):
    @staticmethod
    def forward(var, input, weight, bias, device):

        var.save_for_backward(input, weight, bias)
        var.device = device

        output = gp_Matrix_multiplication(input, weight, input.size(0), weight.size(1), device)

        output+=bias

        return output
    
    def backward(var, gradient_out):
        input, weight, bias = var.saved_tensors
        device = var.device

        trans_weight = gp_Trans_pose(weight, weight.size(1), weight.size(0), device)

        gradient_in = gp_Matrix_multiplication(gradient_out, trans_weight, gradient_out.size(0), trans_weight.size(1), device)

        trans_input = gp_Trans_pose(input, input.size(1), input.size(0), device)

        gradient_weight = gp_Matrix_multiplication(trans_input, gradient_out, trans_input.size(0), gradient_out.size(1), device)

        gradient_bias = gradient_out.sum(0)

        return gradient_in, gradient_weight, gradient_bias, None

class FhLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(FhLinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.device = device

        nn.init.xavier_uniform_(self.weight)
        nn.init.uniform_(self.bias)

    def forward(self, input):
        return FhLinearFunction.apply(input, self.weight, self.bias, self.device)
