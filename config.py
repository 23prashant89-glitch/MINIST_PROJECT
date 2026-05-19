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
    