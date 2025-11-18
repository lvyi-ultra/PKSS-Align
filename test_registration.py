#!/usr/bin/env python3
"""
点云配准测试脚本
使用 Open3D 库演示点云配准流程，对比 PKSS-Align 项目中的测试数据
"""

import open3d as o3d
import numpy as np
import copy
import time
import os

def load_point_cloud(filepath):
    """加载点云文件"""
    print(f"加载点云: {filepath}")
    pcd = o3d.io.read_point_cloud(filepath)
    print(f"  点数: {len(pcd.points)}")
    return pcd

def preprocess_point_cloud(pcd, voxel_size):
    """点云预处理：下采样和计算法线、特征"""
    print(f"下采样，体素大小: {voxel_size}")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # 估计法线
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # 计算FPFH特征
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """全局配准：使用RANSAC"""
    distance_threshold = voxel_size * 1.5
    print(f"全局配准 (RANSAC)，距离阈值: {distance_threshold}")

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    return result

def execute_icp_registration(source, target, init_transform, voxel_size):
    """精细配准：使用ICP"""
    distance_threshold = voxel_size * 0.4
    print(f"ICP精细配准，距离阈值: {distance_threshold}")

    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    return result

def draw_registration_result(source, target, transformation, title="配准结果"):
    """可视化配准结果"""
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # 黄色 - 源点云
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # 蓝色 - 目标点云
    source_temp.transform(transformation)

    print(f"\n{title}")
    print("黄色 = 源点云（已变换）, 蓝色 = 目标点云")

    # 保存可视化结果
    combined = source_temp + target_temp
    return combined

def compute_registration_metrics(source, target, transformation):
    """计算配准质量指标"""
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)

    # 计算点到点距离
    distances = source_temp.compute_point_cloud_distance(target)
    distances = np.asarray(distances)

    mse = np.mean(distances ** 2)
    rmse = np.sqrt(mse)
    mean_dist = np.mean(distances)
    max_dist = np.max(distances)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'Mean Distance': mean_dist,
        'Max Distance': max_dist
    }

def run_registration_pipeline(source_path, target_path, voxel_size=0.05):
    """运行完整的配准流程"""
    print("=" * 60)
    print("点云配准测试")
    print("=" * 60)

    # 加载点云
    source = load_point_cloud(source_path)
    target = load_point_cloud(target_path)

    # 预处理
    print("\n--- 预处理 ---")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    print(f"下采样后点数 - 源: {len(source_down.points)}, 目标: {len(target_down.points)}")

    # 全局配准
    print("\n--- 全局配准 (RANSAC + FPFH) ---")
    start_time = time.time()
    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    ransac_time = time.time() - start_time

    print(f"RANSAC 结果:")
    print(f"  适应度 (fitness): {result_ransac.fitness:.4f}")
    print(f"  内点RMSE: {result_ransac.inlier_rmse:.6f}")
    print(f"  耗时: {ransac_time:.3f}s")

    # ICP精细配准
    print("\n--- ICP 精细配准 ---")

    # 为ICP估计法线
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    start_time = time.time()
    result_icp = execute_icp_registration(
        source, target, result_ransac.transformation, voxel_size)
    icp_time = time.time() - start_time

    print(f"ICP 结果:")
    print(f"  适应度 (fitness): {result_icp.fitness:.4f}")
    print(f"  内点RMSE: {result_icp.inlier_rmse:.6f}")
    print(f"  耗时: {icp_time:.3f}s")

    # 计算配准质量
    print("\n--- 配准质量评估 ---")
    metrics = compute_registration_metrics(source, target, result_icp.transformation)
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    # 变换矩阵
    print("\n--- 最终变换矩阵 ---")
    print(result_icp.transformation)

    # 保存结果
    print("\n--- 保存结果 ---")

    # 保存配准后的点云
    source_aligned = copy.deepcopy(source)
    source_aligned.transform(result_icp.transformation)

    output_dir = os.path.dirname(source_path)
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_aligned_open3d.ply")
    o3d.io.write_point_cloud(output_path, source_aligned)
    print(f"配准后的点云已保存到: {output_path}")

    # 保存合并的可视化
    combined = draw_registration_result(source, target, result_icp.transformation)
    combined_path = os.path.join(output_dir, f"{base_name}_combined_result.ply")
    o3d.io.write_point_cloud(combined_path, combined)
    print(f"合并可视化已保存到: {combined_path}")

    print("\n" + "=" * 60)
    print("配准完成!")
    print(f"总耗时: {ransac_time + icp_time:.3f}s")
    print("=" * 60)

    return result_icp.transformation, metrics

def main():
    # 数据路径
    data_dir = "/home/user/PKSS-Align/PKSS_Align_EXE/Data"

    # 测试飞机数据
    print("\n" + "#" * 60)
    print("# 测试 1: 飞机点云配准 (ModelNet40)")
    print("#" * 60)

    airplane_source = os.path.join(data_dir, "airplane_0001_s.ply")
    airplane_target = os.path.join(data_dir, "airplane_0001_t.ply")

    if os.path.exists(airplane_source) and os.path.exists(airplane_target):
        transform1, metrics1 = run_registration_pipeline(
            airplane_source, airplane_target, voxel_size=0.02)
    else:
        print("飞机数据文件不存在!")

    # 测试室内场景数据
    print("\n" + "#" * 60)
    print("# 测试 2: 室内场景点云配准 (S3DIS)")
    print("#" * 60)

    scene_source = os.path.join(data_dir, "01f_s.ply")
    scene_target = os.path.join(data_dir, "01f_t.ply")

    if os.path.exists(scene_source) and os.path.exists(scene_target):
        transform2, metrics2 = run_registration_pipeline(
            scene_source, scene_target, voxel_size=0.05)
    else:
        print("室内场景数据文件不存在!")

    print("\n\n配准测试全部完成!")
    print("可以使用 MeshLab 或 CloudCompare 等工具查看生成的 .ply 文件")

if __name__ == "__main__":
    main()
