import torch
from torch import nn
import numpy as np

from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.encoder import *
from utils.model import Model

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

path_to_model = "/home/willma32/radiation_control/models/lightning_logs/version_11/checkpoints/epoch=249-step=131499.ckpt"
path_to_clouds = "/bigdata/hplsim/aipp/Anna/FEL/electron_clouds"
path_to_radiation_gt = "/bigdata/hplsim/aipp/Anna/FEL/radiation_simulations"

electron_cloud = np.load(path_to_clouds + '/Track_1.81e-08_1.81e-08_0.0024_0.0024.npy')
radiation_gt = np.load(path_to_radiation_gt + '/radiation_gt_1.81e-08_1.81e-08_0.0024_0.0024.npy')

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = Model.load_from_checkpoint(path_to_model)
model.eval()

vmin = torch.load('./normalization/vmin_outputs.pt').cpu().detach().numpy()
vmax = torch.load('./normalization/vmax_outputs.pt').cpu().detach().numpy()
vmin_p = torch.load('./normalization/vmin_params.pt')
vmax_p = torch.load('./normalization/vmax_params.pt')

pars = np.delete(electron_cloud[:,4:-1], -2, axis=1)

vmin_p_npy = vmin_p.detach().cpu().numpy()
vmax_p_npy = vmax_p.detach().cpu().numpy()

#compute radiation for particles, that can emmit
pars = pars[(pars[:,0] <= vmax_p_npy[0]) & (pars[:,0] >= vmin_p_npy[0])
           & (pars[:,1] <= vmax_p_npy[1]) & (pars[:,1] >= vmin_p_npy[1])
           & (pars[:,2] <= vmax_p_npy[2]) & (pars[:,2] >= vmin_p_npy[2])
           & (pars[:,3] <= vmax_p_npy[3]) & (pars[:,3] >= vmin_p_npy[3])
           & (pars[:,4] <= vmax_p_npy[4]) & (pars[:,4] >= vmin_p_npy[4])]

pars = torch.from_numpy(pars).float()

convert_params = {'int_distance': 20, 'num_pars': 5, 'threshold': 0.05}

pars_norm = (pars - vmin_p) / (vmax_p - vmin_p)
pred_p, pred_v = model(pars_norm.float())

pred_p = (pred_p * (model.pmax - model.pmin) + model.pmin).cpu().detach().numpy()
pred_v = pred_v.cpu().detach().numpy()

radiation = np.zeros((101,200))

for i in range(pred_v.shape[0]):
    x_ = decode(pred_p[i,:,:], pred_v[i,:,:])
    x_ = (x_ * (vmax - vmin) + vmin)
    radiation = radiation + x_
    
print(radiation)