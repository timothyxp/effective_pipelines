def count_zero_grads(optimizer):
    zero_params = 0
    all_params = 0

    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            all_params += param.numel()
            zero_params += (param.grad == 0).sum().detach().cpu().numpy()
    
    return zero_params / all_params
