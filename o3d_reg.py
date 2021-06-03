import open3d as o3d
import numpy as np
import copy
from utils import *



def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    source_temp_np = np.asarray(source_temp.points)
    target_temp_np = np.asarray(target_temp.points)
    mean_dist = np.linalg.norm(np.asarray(source_temp.points) - target_temp_np ,axis=-1).mean()
    print(f"Mean dist:{mean_dist}")
    save_las(source_temp_np[:,:3],'source.las')
    save_las(target_temp_np[:,:3],'target.las')
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0., -0.2951, 1.],
                                      lookat=[0., 0., 0.],
                                      up=[0., -0.9189, 1.])
    

voxel_size = 0.05
threshold = voxel_size * 0.4
cloud = torch.from_numpy(load_las(r'/media/raid/sam/ams_dataset/5D74EOHE.laz')).float().cpu().numpy().astype(np.float32)
center = cloud[:,:2].mean(axis=0)
cloud[:,:2] = cloud[:,:2] - center 
source_cloud = cloud[extract_area(cloud,cloud[:,:2].mean(axis=0),10),:]
target_cloud = source_cloud.copy() + np.array([0.1,0.1,0,0,0,0]).astype(np.float32)


source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(source_cloud[:,:3])

source = source.voxel_down_sample(voxel_size)
target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(target_cloud[:,:3])

target = target.voxel_down_sample(voxel_size)
trans_init = np.eye(4).astype(np.float32)
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