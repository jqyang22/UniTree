import torch.optim as optim

def adaDelta(params):
    return optim.Adadelta(params, lr=1.5, rho=0.95, eps=1e-6)

def adam(params):
    return optim.Adam(params, lr=0.0001, betas=(0.9, 0.999), eps=1.0e-8)

def nadam(params):
    return optim.Adam(params, lr=0.002, betas=(0.9, 0.999))

def adagrad(params):
    return optim.Adagrad(params, lr=0.01)
