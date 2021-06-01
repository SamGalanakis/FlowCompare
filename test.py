from dataloaders import AmsGridLoaderPointwise




dataset=AmsGridLoaderPointwise('/media/raid/sam/ams_dataset',out_path='save/processed_dataset',preload=False,subsample='fps',sample_size=1024,min_points=100,grid_type='circle',normalization='co_unit_sphere',grid_square_size=1)