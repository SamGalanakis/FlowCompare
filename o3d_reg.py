import open3d as o3d
import numpy as np
import copy
from utils import *
from sklearn.neighbors import KDTree
from knn import KNN_torch


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    

    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0., -0.2951, 1.],
                                      lookat=[0., 0., 0.],
                                      up=[0., -0.9189, 1.])
    


if __name__ == '__main__':
    voxel_size = 0.07
    threshold = voxel_size * 0.4
    source_cloud = torch.from_numpy(load_las(r'/media/raid/sam/ams_dataset/WE1QUNRO.laz')).float().cpu().numpy().astype(np.float32)
    target_cloud = torch.from_numpy(load_las(r'/media/raid/sam/ams_dataset/5D6TROEX.laz')).float().cpu().numpy().astype(np.float32)
    center = source_cloud[:,:2].mean(axis=0)
    source_cloud[:,:2] = source_cloud[:,:2] - center 
    target_cloud[:,:2] = target_cloud[:,:2] - center 


    trans_init = np.eye(4).astype(np.float32)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_cloud[:,:3])
    source = source.voxel_down_sample(voxel_size)

    source_torch = torch.from_numpy(np.asarray(source.points))
    sample_points = random_subsample(source_torch,100)
    knn = KNN_torch(1000)
    fitted_knn,_ = knn(source_torch)
    indices,_ = fitted_knn(sample_points)
    closest = source_torch[indices.reshape(-1),:]
    closest_geom = o3d.geometry.PointCloud()
    closest_geom.points = o3d.utility.Vector3dVector(closest)
    draw_registration_result(source,closest_geom,trans_init)

    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_cloud[:,:3])

    target = target.voxel_down_sample(voxel_size)

    #draw_registration_result(source, target, trans_init)

    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)
    target.estimate_normals()
    source.estimate_normals()
    print("Apply point-to-plane ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100000))
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    draw_registration_result(source, target, reg_p2l.transformation)