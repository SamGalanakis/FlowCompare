
import torch
import numpy as np
import open3d as o3d


def context_voxel_center(voxel):
    voxel = voxel[:,:3]
    _min = voxel.min(dim=0)[0]
    _max = voxel.max(dim=0)[0]
    approx_center = _min/2 + _max/2
    return approx_center


def icp_reg_precomputed_target(source_cloud, target, voxel_size=0.05, max_it=2000):
    source_cloud = source_cloud.cpu().numpy()
    
    threshold = voxel_size * 0.4
    trans_init = np.eye(4).astype(np.float32)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_cloud[:, :3])
    source.colors = o3d.utility.Vector3dVector(source_cloud[:, 3:])
    source = source.voxel_down_sample(voxel_size)
    source.estimate_normals()
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_it))

    return result

def downsample_transform(cloud, voxel_size, transform):
    device = cloud.device
    source = o3d.geometry.PointCloud()
    cloud = cloud.cpu().numpy()
    source.points = o3d.utility.Vector3dVector(cloud[:, :3])
    source.colors = o3d.utility.Vector3dVector(cloud[:, 3:])
    source = source.voxel_down_sample(voxel_size)
    source.transform(transform)
    cloud = torch.from_numpy(np.concatenate(
        (np.asarray(source.points), np.asarray(source.colors)), axis=-1))
    return cloud.to(device)




def registration_pipeline(cloud_list,voxel_size_registration,
    voxel_size_final):
    '''Takes each entry in cloud_list which is a list of clouds
    , registers each cloud to the first entry and then returns
     downsampled versions'''
    
    
    device = cloud_list[0].device
    # Apply registration between each cloud and first in list, store transforms
    # First cloud does not need to be transformed
    registration_transforms = [np.eye(4, dtype=np.float32)]
    
    
    
    common_center  =  cloud_list[0].mean(axis=0)
    common_center[3:] =0.0
    cloud_list_ = [x - common_center for x in cloud_list]
    target_cloud = cloud_list_[0].cpu().numpy()
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_cloud[:, :3])
    target.colors = o3d.utility.Vector3dVector(target_cloud[:, 3:])
    target = target.voxel_down_sample(voxel_size_registration)
    target.estimate_normals()
    for source_cloud in cloud_list[1:]:
                    result = icp_reg_precomputed_target(
                        source_cloud, target, voxel_size=voxel_size_registration)
                    registration_transforms.append(result.transformation)
    # Downsample and apply registration
    cloud_list = [downsample_transform(x, voxel_size_final, transform) for x, transform in zip(
                    cloud_list, registration_transforms)]

    cloud_list = [x.to(device) for x in cloud_list]
    return cloud_list

