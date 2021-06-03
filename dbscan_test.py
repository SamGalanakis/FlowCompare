import cupoch as cph
import numpy as np
from utils import load_las,save_las,extract_area,random_subsample,grid_split,view_cloud_plotly,view_cloud_o3d,co_min_max
import sys
import torch
from models.pct_utils import query_ball_point
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.io as pio
import webbrowser



def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        cph.geometry.KDTreeSearchParamRadius(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = cph.registration.compute_fpfh_feature(
        pcd,
        cph.geometry.KDTreeSearchParamRadius(radius=radius_feature, max_nn=30))
    return pcd, pcd_fpfh







def register(source_cloud,target_cloud,voxel_size,type='ICP'):
    source = cph.geometry.PointCloud()
    target = cph.geometry.PointCloud()
    source.points = cph.utility.Vector3fVector(source_cloud[:,:3])
    source.colors = cph.utility.Vector3fVector(source_cloud[:,3:])
    source  = source.voxel_down_sample(voxel_size)

    target.points = cph.utility.Vector3fVector(target_cloud[:,:3])
    target.colors = cph.utility.Vector3fVector(target_cloud[:,3:])
    target  = target.voxel_down_sample(voxel_size)
    # transform = np.random.rand(4,4)
    # target = target.transform(transform)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    
    #current_transformation = np.identity(4).astype(np.float32)

    

    
    distance_threshold = voxel_size * 0.5
    result = cph.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        cph.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))

    source = cph.geometry.PointCloud()
    target = cph.geometry.PointCloud()
    source.points = cph.utility.Vector3fVector(source_cloud[:,:3])
    source.colors = cph.utility.Vector3fVector(source_cloud[:,3:])
    target.points = cph.utility.Vector3fVector(target_cloud[:,:3])
    target.colors = cph.utility.Vector3fVector(target_cloud[:,3:])
    source.estimate_normals()
    target.estimate_normals()
    result_icp = cph.registration.registration_icp(
        source, target, 0.01,
        init = result.transformation,
        estimation_method = cph.registration.TransformationEstimationPointToPlane(),
        criteria = cph.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                    relative_rmse=1e-6,
                                                    max_iteration=50))


    st = target.transform(result_icp.transformation)
    st = np.asarray(st.points.cpu())
    
    src = np.asarray(source.points.cpu())
    src = np.asarray(source.points.cpu())
    save_las(src[:,:3],'src.las')

    save_las(st[:,:3],'target.las')
    return result.transformation


sys.path.append(r'C:\on_path')

voxel_size=0.12
cloud = torch.from_numpy(load_las(r'/media/raid/sam/ams_dataset/5D74EOHE.laz')).float().cpu().numpy()
center = cloud[:,:2].mean(axis=0)
cloud[:,:2] = cloud[:,:2] - center 
cloud = cloud[extract_area(cloud,cloud[:,:2].mean(axis=0),10),:]
cloud[:,:3] = cloud[:,:3] - cloud.max(axis=0)[:3]
target_cloud = cloud.copy() + np.array([0.5,0.05,0.02,0,0,0]).astype(np.float32)



result = register(cloud,target_cloud,voxel_size=voxel_size)
source = cph.geometry.PointCloud()
target = cph.geometry.PointCloud()
source.points = cph.utility.Vector3fVector(cloud[:,:3])
source.colors = cph.utility.Vector3fVector(cloud[:,3:])
source  = source.voxel_down_sample(voxel_size)

target.points = cph.utility.Vector3fVector(target_cloud[:,:3])
target.colors = cph.utility.Vector3fVector(target_cloud[:,3:])
target  = target.voxel_down_sample(voxel_size)
source = source.v
source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

result = execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size)













