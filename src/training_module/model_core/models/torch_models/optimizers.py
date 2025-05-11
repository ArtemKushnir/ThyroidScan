from torch.optim import SGD, Adagrad, Adam, AdamW, RMSprop

OPTIMIZER = {"sgd": SGD, "adam": Adam, "rms_prop": RMSprop, "adagrad": Adagrad, "adamw": AdamW}
