from dataloaders import AmsGridLoaderPointwise
from tqdm import tqdm

from torch.utils.data import DataLoader

dataset=AmsGridLoaderPointwise('/media/raid/sam/ams_dataset',out_path='save/processed_dataset',preload=True,n_samples=1024,device='cpu')
dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=4, collate_fn=lambda x : x, pin_memory=True, prefetch_factor=2, drop_last=True)
for batch in tqdm(dataloader):  
    pass
pass