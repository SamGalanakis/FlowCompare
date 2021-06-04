from dataloaders import AmsGridLoaderPointwise




dataset=AmsGridLoaderPointwise('/media/raid/sam/ams_dataset',out_path='save/processed_dataset',preload=False,min_points=800,grid_square_size=2,device='cpu')