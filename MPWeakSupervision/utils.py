
def print_parameter_count(model):
    """
    https://stackoverflow.com/a/62508086/198401
    """
    
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        print(f"{name}\t{params}")
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params
