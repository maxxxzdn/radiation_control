import torch
from torch.utils.data import Dataset
import numpy as np

def get_vmin_vmax_rad(items, log_=False):
    print(len(items))
    for ind, item in enumerate(items):
        if ind%1000==0:
            print(ind)
        #par, arr = get_file_radiation(ind, items)
        arr = np.load(items[ind])
        if log_:
            arr = np.log10(arr)
        par = [float(p) for p in items[ind].split('.npy')[0].split('/')[-1].split('_')[1:]]
        
        if item == items[0]:
            vmin = np.min(arr)
            vmax = np.max(arr)

            vmin_p = [par[i] for i in range(len(par))]
            vmax_p = [par[i] for i in range(len(par))]
        else:
            vmin = min(np.min(arr), vmin)
            vmax = max(np.max(arr), vmax)

            vmin_p = [min(par[i], vmin_p[i]) for i in range(len(par))]
            vmax_p = [max(par[i], vmax_p[i]) for i in range(len(par))]
    return  torch.Tensor(vmin_p).float(), torch.Tensor(vmax_p).float(), torch.torch.full(arr.shape, vmin),  torch.torch.full(arr.shape, vmax)

class PCDataset(Dataset):
    def __init__(self, items, get_data, num_files=-1, normalize=False, log_=False):
        self.get_data = get_data
        self.normalize = normalize
        self.log_ = log_
        if num_files > 0:
            self.items = items[:num_files]
        else:
            self.items = items
        
        #self.vmin_p, self.vmax_p, self.vmin, self.vmax = get_vmin_vmax_rad(items, log_)
      
        print('Total number of files: ', len(self.items))

    def __getitem__(self, index):
        return self.get_data(ind=index, items=self.items, log_=self.log_)

    def __len__(self):
        return len(self.items)