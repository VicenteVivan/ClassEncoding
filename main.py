from sched import scheduler


if __name__ == '__main__':
    import os, numpy as np, argparse, time
    from tqdm import tqdm

    import torch
    import torch.nn as nn

    import dataloader
    from train_and_eval import train, evaluate
    
    import wandb

    import models 
    from config import getopt
    from models import ResNet50
    
    from torchvision import datasets, transforms
    from dataloader import train_dataset, val_dataset

    opt = getopt()

    config = {
        'learning_rate' : opt.lr,
        'epochs' : opt.n_epochs,
        'batch_size' : opt.batch_size,
        'architecture' : opt.archname
        }
    
    # Load CIFAR100
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_dataloader  = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    
    w = wandb.init(project='CIFAR100 ConEmb',
                    entity='vicentevivan',
                    config=config)
    
    wandb.run.name = opt.description
    
    model = ResNet50()
    model = model.to(opt.device)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    _ = model.to(opt.device)

    wandb.watch(model, criterion, log="all")

    for epoch in range(opt.n_epochs):
        evaluate(val_dataloader=val_dataloader, model=model, model_name=opt.archname, criterion=criterion, epoch=epoch, opt=opt)
        if not opt.evaluate:
            _ = model.train()
            loss = train(train_dataloader=train_dataloader, model=model, model_name=opt.archname, criterion=criterion, optimizer=optimizer, opt=opt, epoch=epoch)
            scheduler.step()
        
    del model
    w.finish()
