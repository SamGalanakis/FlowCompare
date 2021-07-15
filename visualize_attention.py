import torch
from utils import view_cloud_plotly
import os
import random

out_path = 'save/figs'
extract_0 = torch.load('save/for_visualizations/extract_0.pt').cpu()
extract_1 = torch.load('save/for_visualizations/extract_1.pt').cpu()
weights_aug = torch.load('save/for_visualizations/weights.pt').cpu()
weights_50 = torch.load('save/for_visualizations/weights_50.pt').cpu()
weights_110 = torch.load('save/for_visualizations/weights_110.pt').cpu()
pick_points = 2

weight_names = ['aug','50','110']



batch_size = extract_0.shape[0]


for batch_ind in range(batch_size):
    c_0 = extract_0[batch_ind]
    c_1 = extract_1[batch_ind]
    w_aug = weights_aug[batch_ind]
    w_50 = weights_50[batch_ind]
    w_100 = weights_110[batch_ind]

    weights =[weights_aug[batch_ind],weights_50[batch_ind],weights_110[batch_ind]]

    random_point_indices = torch.randint(0,weights[0].shape[0],(pick_points,))
    for index in random_point_indices.tolist():

        for weight_name,w in zip(weight_names,weights):
            w_ = w[index]
            w_ = (w_-w_.min())/(w_.max()-w_.min())
            point_1 = c_1[index].unsqueeze(0)
            c_0 = torch.cat((c_0[:,:3],w_.unsqueeze(-1)),dim=-1)
            point_1 = torch.cat((point_1[:,:3],torch.tensor([1]).unsqueeze(0).float()),dim=-1)
            
            
            fig = view_cloud_plotly(c_0[:,:3],c_0[:,3:],show=False,colorscale='Hot',show_scale=True,point_size=5)
            fig = view_cloud_plotly(point_1,torch.tensor([0,1,0]).float().unsqueeze(0),fig=fig,point_size=10,show=False)
            fig.write_html(os.path.join(out_path,f'test_{batch_ind}_{index}_{weight_name}.html'))


