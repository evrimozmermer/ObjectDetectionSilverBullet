import collections
import numpy as np
import os
import torch
import tqdm

import config
import datasets
import optimizers
from models import retinanet
from utils import *

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))
cfg = config.load("./config/config.json")
start_epoch = 0
cfg.resume = "./checkpoints/{0}-{1}-{2}-W{3}H{4}-InpSize{5}-{6}-parallel{7}/ckpt".format(cfg.dataset, 
                                                          "RetinaNet",
                                                          cfg.backbone,
                                                          cfg.width,
                                                          cfg.height,
                                                          cfg.input_size,
                                                          cfg.optimizer,
                                                          cfg.data_parallel) # cfg.__dict__["dataset"]

try:
    os.mkdir("/".join(cfg.resume.split("/")[:3]))
except:
    pass

# import dataset
os.chdir("datasets")
cfg.data_root = os.getcwd()
dl_tr, dl_ev = datasets.load(cfg, val=True)
os.chdir("..")

model = retinanet.load(cfg)

# resume
checkpoint = None
if os.path.isfile("{}.pth".format(cfg.resume)):
    print('=> loading checkpoint:\n{}.pth'.format(cfg.resume))
    checkpoint = torch.load("{}.pth".format(cfg.resume),torch.device(cfg.device))
    if checkpoint['parallel_flag']:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['model_state_dict'], strict = True)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict = True)
    start_epoch = checkpoint['epoch']

if cfg.data_parallel and not checkpoint['parallel_flag']:
    model = torch.nn.DataParallel(model)
model.to(cfg.device)
model.training = True

param_groups = model.parameters()
opt = optimizers.load(cfg, param_groups)
if checkpoint:
    opt.load_state_dict(checkpoint['optimizer'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, verbose=True)
loss_hist = collections.deque(maxlen=20)

for epoch in range(start_epoch, cfg.epochs):

    model.train()
    if cfg.data_parallel:
        model.module.freeze_bn()
    else:
        model.freeze_bn()

    epoch_loss = []

    pbar = tqdm.tqdm(enumerate(dl_tr, start = 1))
    for iter_num, data in pbar:

        if torch.cuda.is_available():
            classification_loss, regression_loss = model([data['image'].cuda().float(), data['annot'].cuda().float()])
        else:
            classification_loss, regression_loss = model([data['image'].float(), data['annot']])
            
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        loss = classification_loss + regression_loss
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        opt.step()

        loss_hist.append(float(loss))
        epoch_loss.append(float(loss))

        print_text = 'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                      epoch, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist))

        pbar.set_description(print_text)
        
        del classification_loss
        del regression_loss

    # scheduler.step(np.mean(epoch_loss))
    
    if (epoch+1)%cfg.visualization_interval==0: 
        visualize(cfg, model, dl_ev)

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
                'epoch': epoch,
                'parallel_flag': cfg.data_parallel},
                '{}.pth'.format(cfg.resume))

model.eval()
torch.save(model, 'model_final.pt')