from dataloaders import AmsVoxelLoader
from tqdm import tqdm

from torch.utils.data import DataLoader

dataset=AmsVoxelLoader('/media/raid/sam/ams_dataset',out_path='save/processed_dataset',preload=True,n_samples=1024,device='cpu')
dataloader = DataLoader(dataset, shuffle=False, batch_size=12, num_workers=0, collate_fn=None, pin_memory=True, prefetch_factor=2, drop_last=True)
for batch in tqdm(dataloader):  
    pass
pass