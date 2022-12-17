import torch

models = {}
def register_model(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def get_model(name, **args):
    net = models[name](**args)
    if torch.cuda.is_available():
        net = net.cuda()
    return net
