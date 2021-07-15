import math
import os
from time import perf_counter
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from dataloaders import AmsVoxelLoader
from utils import is_valid
from model_initialization import inner_loop,initialize_flow,make_sample,load_flow,save_flow
from test_flow import evaluate_on_test





def train(config_path):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    wandb.init(project="flow_change", config=config_path)
    config = wandb.config


    models_dict = initialize_flow(config, device, mode='train')

    
    if config['data_loader'] == 'AmsVoxelLoader':
        dataset = AmsVoxelLoader(config['directory_path_train'],config['directory_path_test'], out_path='save/processed_dataset', preload=config['preload'],
        n_samples = config['sample_size'],final_voxel_size = config['final_voxel_size'],device=device,
        n_samples_context = config['n_samples_context'], context_voxel_size = config['context_voxel_size'],mode='train',getter_mode = config['dataset_get_mode']
        )
     
    else:
        raise Exception('Invalid dataloader type!')


  
    dataloader = DataLoader(dataset, shuffle=True, batch_size=config['batch_size'], num_workers=config[
                            "num_workers"], collate_fn=None, pin_memory=True, prefetch_factor=2, drop_last=True)

    if config["optimizer_type"] == 'Adam':
        optimizer = torch.optim.Adam(
            models_dict['parameters'], lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["optimizer_type"] == 'Adamax':
        optimizer = torch.optim.Adamax(
            models_dict['parameters'], lr=config["lr"], weight_decay=config["weight_decay"], polyak=0.999)
    elif config["optimizer_type"] == 'AdamW':
        optimizer = torch.optim.AdamW(
            models_dict['parameters'], lr=config["lr"], weight_decay=config["weight_decay"])
    elif config['optimizer_type'] == 'SGD':
        optimizer = torch.optim.SGD(models_dict['parameters'], lr=config["lr"],
                                    momentum=0, dampening=0, weight_decay=config["weight_decay"], nesterov=False)
    else:
        raise Exception('Invalid optimizer type!')



    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=  config['patience'], factor=config['lr_factor'],threshold=config['threshold_scheduler'], min_lr=config["min_lr"],verbose=True)


    save_model_path = r'save/conditional_flow_compare'
    loss_running_avg = 0
    # Load checkpoint params if specified path
    if config['load_checkpoint']:
        print(f"Loading from checkpoint: {config['load_checkpoint']}")
        checkpoint_dict = torch.load(config['load_checkpoint'])
        models_dict = load_flow(checkpoint_dict, models_dict)

    else:
        print("Starting training from scratch!")

    # Watch models:
    detect_anomaly = False
    if detect_anomaly:
        print('DETECT ANOMALY ON')
        torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])
    best_so_far = math.inf
    last_save_path = None
    for epoch in range(config["n_epochs"]):
        print(f"Starting epoch: {epoch}")
        for batch_ind, batch in enumerate(tqdm(dataloader)):
            with torch.cuda.amp.autocast(enabled=config['amp']):
                if config['time_stats']:
                    torch.cuda.synchronize()
                    t0 = perf_counter()
                batch = [x.to(device) for x in batch]

                # Set to None if not using
                if not config['using_extra_context']:
                    batch[-1] = None
                extract_0 = batch[0]
                extra_context = batch[-1]


                
                loss, _, nats = inner_loop(
                    batch, models_dict, config)
                is_valid(loss)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                models_dict['parameters'], max_norm=config['grad_clip_val'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(loss) 

            optimizer.zero_grad(set_to_none=True)
            current_lr = optimizer.param_groups[0]['lr']
            if config['time_stats']:
                torch.cuda.synchronize()
                time_batch = perf_counter() - t0
            else:
                time_batch = np.NaN
            loss_item = loss.item()
            loss_running_avg = (
                loss_running_avg*(batch_ind) + loss_item)/(batch_ind+1)



            
            if (batch_ind % config['batches_per_save'])==0 and batch_ind>0:
                if loss_running_avg < best_so_far:
                    if last_save_path!=None:
                        os.remove(last_save_path)
                    print(f'Saving!')
                    savepath = os.path.join(
                        save_model_path, f"{wandb.run.name}_e{epoch}_b{batch_ind}_model_dict.pt")
                    print(f'Loss epoch: {loss_running_avg}')
                    save_flow(models_dict,config,optimizer,scheduler,savepath)
                    last_save_path = savepath
                    best_so_far = min(loss_running_avg,best_so_far)
                    loss_running_avg = 0




            if ((batch_ind+1) % config['batches_per_sample'] == 0) and config['make_samples']:
                with torch.no_grad():
                    
                    cond_nump = extract_0[0].cpu().numpy()
                    if config['using_extra_context']:
                        sample_extra_context = extra_context[0].unsqueeze(0)
                    else : 
                        sample_extra_context = None
                    sample_points = make_sample(
                        4000, extract_0[0].unsqueeze(0), models_dict, config,sample_extra_context)
                    cond_nump[:, 3:6] = np.clip(
                    cond_nump[:, 3:6]*255, 0, 255)
                    sample_points = sample_points.cpu().numpy().squeeze()
                    sample_points[:, 3:6] = np.clip(
                    sample_points[:, 3:6]*255, 0, 255)
                    wandb.log({"Cond_cloud": wandb.Object3D(cond_nump[:, :6]), "Gen_cloud": wandb.Object3D(
                        sample_points[:, :6]), 'loss': loss_item, 'nats': nats.item(), 'lr': current_lr, 'time_batch': time_batch})
            else:
                pass
                wandb.log({'loss': loss_item, 'nats': nats.item(),
                          'lr': current_lr, 'time_batch': time_batch})
            
                
        wandb.log({'epoch': epoch, "loss_epoch": loss_running_avg})
        
        
        
        


if __name__ == "__main__":
    config_path = r"config/extra_300_no_extra_context.yaml"
    train(config_path)
     
    
