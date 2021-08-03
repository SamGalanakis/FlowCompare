from utils import *
import numpy as np
import open3d as o3d


cloud_0_points = load_las('test_cloud_0.las')
cloud_1_points = load_las('test_cloud_3.las')


voxel_size = 0.05
threshold = 0.05 * 0.4
trans_init = np.eye(4).astype(np.float32)
source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(cloud_0_points[:, :3])
source.colors = o3d.utility.Vector3dVector(cloud_0_points[:, 3:])
source = source.voxel_down_sample(voxel_size)
source.estimate_normals()

cloud_1 = o3d.geometry.PointCloud()
cloud_1.points = o3d.utility.Vector3dVector(cloud_1_points[:, :3])
cloud_1.colors = o3d.utility.Vector3dVector(cloud_1_points[:, 3:])
cloud_1 = cloud_1.voxel_down_sample(voxel_size)
cloud_1.estimate_normals()

result = o3d.pipelines.registration.registration_icp(
    source, cloud_1, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
source.transform(result.transformation)


cloud_0 = np.concatenate(
        (np.asarray(source.points), np.asarray(source.colors)), axis=-1)

save_las(cloud_0[:,:3],'test_cloud_0_transformed.las',cloud_0[:,3:])