
import torch
import numpy as np
import open3d as o3d
import copy
def draw_registration_result(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    #source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
def context_voxel_center(voxel):
    """Gives voxel center by taking center of min max for each axis"""
    voxel = voxel[:,:3]
    _min = voxel.min(dim=0)[0]
    _max = voxel.max(dim=0)[0]
    approx_center = _min/2 + _max/2
    return approx_center
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result
def to_o3d(cloud,voxel_size=None):
    o3d_cloud = o3d.geometry.PointCloud()
    cloud = cloud.cpu().numpy()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud[:, :3])
    o3d_cloud.colors = o3d.utility.Vector3dVector(cloud[:, 3:])

    if voxel_size!=None:
        o3d_cloud = o3d_cloud.voxel_down_sample(voxel_size)
    return o3d_cloud

def from_o3d(o3d_cloud):
    cloud = torch.from_numpy(np.concatenate(
        (np.asarray(o3d_cloud.points), np.asarray(o3d_cloud.colors)), axis=-1))
    return cloud

def refine_registration(source, target,voxel_size,pre_transform):
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, pre_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

#     return result

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

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    # print(":: Apply fast global registration with distance threshold %.3f" \
    #         % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def registration_pipeline(cloud_list,voxel_size_registration,
    voxel_size_final):
    '''Takes each entry in cloud_list which is a list of clouds
    , registers each cloud to the first entry and then returns
     downsampled versions'''
    
    
    device = cloud_list[0].device
    # Apply registration between each cloud and first in list, store transforms
    # First cloud does not need to be transformed
    registration_transforms = [np.eye(4, dtype=np.float32)]
    
    #trans_init = np.eye(4).astype(np.float32)
    
    common_center  =  cloud_list[0].mean(axis=0)
    common_center[3:] =0.0
    cloud_list = [x - common_center for x in cloud_list]
    
    cloud_list_down = [to_o3d(x,voxel_size= voxel_size_registration) for x in cloud_list]
    radius_normal = voxel_size_registration * 2
    for cloud_down in cloud_list_down:
        cloud_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size_registration * 5
    pcd_fpfh_list_down =  [ o3d.pipelines.registration.compute_fpfh_feature(
        x,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))  for x in cloud_list_down] 
    target_down = cloud_list_down[0]
    
    
    print('Global registration!')
    for source_cloud,source_pcd_fpfh in zip(cloud_list_down[1:],pcd_fpfh_list_down[1:]):
                    result = execute_global_registration(source_cloud,target_down,source_pcd_fpfh,pcd_fpfh_list_down[0],voxel_size_registration)
                    registration_transforms.append(result.transformation)
    voxel_size_icp = 0.02
    radius_feature = 5 *voxel_size_icp 
    cloud_list = [to_o3d(x,voxel_size= voxel_size_icp) for x in cloud_list]
    for cloud in cloud_list:
        cloud.estimate_normals()
    # pcd_fpfh_list = [ o3d.pipelines.registration.compute_fpfh_feature(
    #     x,
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))  for x in cloud_list] 
    final_transforms = [np.eye(4, dtype=np.float32)]
    print('Refinement!')
    for  source_cloud,pre_transform in zip(cloud_list[1:],registration_transforms[1:]):
                    result = refine_registration(source_cloud,cloud_list[0],voxel_size_icp,pre_transform)
                    final_transforms.append(result.transformation)

    cloud_list =[x.voxel_down_sample(voxel_size_final) for x in cloud_list]
    for x,transform in zip(cloud_list,final_transforms):
        x.transform(transform)
    
    cloud_list = [from_o3d(x).to(device) for x in cloud_list]
    return cloud_list

