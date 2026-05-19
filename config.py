import torch
class Config:
    #data
    bacth_size = 64
    num_workers = 2


    # model
    input_size = 784
    hidden_size = [256, 128]
    num_classes = 10
    dropout = 0.2


    # training
    max_epoch = 50
    learning_rate = 0.001
    weight_decay = 1e-4

    # shedular 
    shedular_patience = 3
    shedular_factor = 0.5

    #early stopping
    early_stop_patience = 7

    #path
    checkpoint_dir = 'checkpoint'
    log_dir = 'logs'

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'