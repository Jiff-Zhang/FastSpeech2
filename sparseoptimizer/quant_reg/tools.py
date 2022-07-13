import torch

scale_dict = {}

def qr_loss_log(quant_reg_losses: [torch.FloatTensor]):
    log = ""
    for loss in quant_reg_losses:
        log += f", {loss.item()}"
    return log

def true_softmax(quant_reg_losses: [torch.FloatTensor]):
    max_quant_reg_losses = max(quant_reg_losses).item()
    for loss in quant_reg_losses:
        loss.sub_(max_quant_reg_losses)
        loss.exp_()
    quant_reg_loss = sum(quant_reg_losses) 
    quant_reg_loss.log_()
    return quant_reg_loss 

def get_weight_scale(weight, num_bits=8):
    if len(weight.shape) == 2:
        max_value = weight.abs().max(-1)[0]
    else:
        print("Error: Invalid input shape {}".format(weight.shape))
        raise
    scale = max_value/(2**(num_bits-1)-1)
    return scale

def get_input_scale(input_tensor, key=None, num_bits=8):
    global scale_dict
    if key:
        if key in scale_dict:
            scale = scale_dict[key]
        else:
            scale = input_tensor.abs().max()/(2**(num_bits-1)-1)
            scale_dict[key] = scale 
            print(key, scale)
    else:
        scale = input_tensor.abs().max()/(2**(num_bits-1)-1) 
    return scale

def quant_data(x, scale):
    int32_x = (x / scale).clip(-128, 127).round().int()
    return int32_x

def dequant_data(x, scale):
    return x.float() * scale

if __name__ == "__main__":
    A = torch.FloatTensor([0.2])
    B = torch.FloatTensor([0.2])
    print(true_softmax([A, B]))