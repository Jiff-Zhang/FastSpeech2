import torch
from torch import nn
import torch.nn.functional as F

from .tools import (
    get_input_scale,
    get_weight_scale,
    quant_data,
    dequant_data
)

def quant_reg_loss(
    x: torch.FloatTensor, 
    module: nn.Module, 
    add_term: torch.FloatTensor = None,
    output_y: bool = False):

    A_max = get_input_scale(x, )
    A_max.detach()
    A_max.log_()

    W_max = get_weight_scale(module.weight)
    W_max[W_max == 0] = 1
    W_max.log_()

    y = module(x)
    if add_term is not None:
        C = y + add_term
    else:
        C = y 

    C_max = get_input_scale(C)
    C_max.log_() 

    m1_loss = C_max - A_max - W_max - 1
    m1_loss[m1_loss < 0] = 0
    m1_loss.square_()
    m1_loss = m1_loss.mean() 

    if output_y:
        return m1_loss, y
    else:
        return m1_loss

def quant_dequant_output(
    x: torch.FloatTensor, 
    y: torch.FloatTensor, 
    module: nn.Module = None, 
    weight: torch.FloatTensor = None,
    bias: torch.FloatTensor = None,
    add_term: torch.FloatTensor = None):

    if not weight and module:
        weight = module.weight

    if not bias and module:
        bias = module.bias

    scales_x = get_input_scale(x, key=f"{id(module)}_x")
    # scales_x = get_input_scale(x)
    scales_weight = get_weight_scale(weight)
    int32_x = quant_data(x, scales_x)

    if bias is not None:
        int32_bias = bias / (scales_x * scales_weight)
    else:
        int32_bias = torch.tensor(0, dtype=torch.float32)

    int32_bias = int32_bias.int()
    int32_weight = quant_data(weight, scales_weight.unsqueeze(-1))

    if add_term is not None:
        scales_add_term = get_input_scale(add_term, key=f"{id(module)}_add")
        # scales_add_term = get_input_scale(add_term)
        int32_add_term = quant_data(add_term, scales_add_term)
        M2 = (scales_add_term / scales_x / scales_weight).to(torch.float32)
    else:
        int32_add_term = torch.tensor(0, dtype=torch.int32)
        M2 = torch.tensor(1, dtype=torch.float32)

    scales_y = get_input_scale(y, key=f"{id(module)}_y")
    # scales_y = get_input_scale(y)

    M1 = (scales_x * scales_weight / scales_y).to(torch.float32)
    quant_y = M1 * (F.linear(int32_x.float(), int32_weight.float(), int32_bias.float()) + M2 * int32_add_term.float())
    quant_y = torch.clip(quant_y.round(), -128, 127).to(torch.int8)
    dequant_y = dequant_data(quant_y, scales_y)

    return dequant_y

def quant_dequant_output_matmal(
    x1: torch.FloatTensor, 
    x2: torch.FloatTensor, 
    y: torch.FloatTensor,
    module: nn.Module
):
    scales_x1 = get_input_scale(x1, key=f"{id(module)}_x1")
    scales_x2 = get_input_scale(x2, key=f"{id(module)}_x2")
    # scales_x1 = get_input_scale(x1)
    # scales_x2 = get_input_scale(x2)

    int32_x1 = quant_data(x1, scales_x1)
    int32_x2 = quant_data(x2, scales_x2)

    scales_y = get_input_scale(y, key=f"{id(module)}_y")
    # scales_y = get_input_scale(y) 
    
    M1 = (scales_x1 * scales_x2 / scales_y).to(torch.float32)
    quant_y = M1 * torch.matmul(int32_x1.float(), int32_x2.float())
    quant_y = torch.clip(quant_y.round(), -128, 127).to(torch.int8)
    dequant_y = dequant_data(quant_y, scales_y)

    return dequant_y


if __name__ == "__main__":
    pass