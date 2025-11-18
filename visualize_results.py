#!/usr/bin/env python3
"""
可视化配准结果，生成2D投影图片
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_registration(source_path, target_path, aligned_path, output_image, title):
    """生成配准前后的对比图"""

    # 加载点云
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)
    aligned = o3d.io.read_point_cloud(aligned_path)

    # 获取点坐标
    source_pts = np.asarray(source.points)
    target_pts = np.asarray(target.points)
    aligned_pts = np.asarray(aligned.points)

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=14)

    # 配准前 - XY视图
    axes[0, 0].scatter(source_pts[:, 0], source_pts[:, 1], s=0.1, c='red', alpha=0.5, label='Source')
    axes[0, 0].scatter(target_pts[:, 0], target_pts[:, 1], s=0.1, c='blue', alpha=0.5, label='Target')
    axes[0, 0].set_title('Before Registration (XY)')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].legend(markerscale=10)
    axes[0, 0].axis('equal')

    # 配准前 - XZ视图
    axes[0, 1].scatter(source_pts[:, 0], source_pts[:, 2], s=0.1, c='red', alpha=0.5)
    axes[0, 1].scatter(target_pts[:, 0], target_pts[:, 2], s=0.1, c='blue', alpha=0.5)
    axes[0, 1].set_title('Before Registration (XZ)')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    axes[0, 1].axis('equal')

    # 配准前 - YZ视图
    axes[0, 2].scatter(source_pts[:, 1], source_pts[:, 2], s=0.1, c='red', alpha=0.5)
    axes[0, 2].scatter(target_pts[:, 1], target_pts[:, 2], s=0.1, c='blue', alpha=0.5)
    axes[0, 2].set_title('Before Registration (YZ)')
    axes[0, 2].set_xlabel('Y')
    axes[0, 2].set_ylabel('Z')
    axes[0, 2].axis('equal')

    # 配准后 - XY视图
    axes[1, 0].scatter(aligned_pts[:, 0], aligned_pts[:, 1], s=0.1, c='orange', alpha=0.5, label='Aligned')
    axes[1, 0].scatter(target_pts[:, 0], target_pts[:, 1], s=0.1, c='blue', alpha=0.5, label='Target')
    axes[1, 0].set_title('After Registration (XY)')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].legend(markerscale=10)
    axes[1, 0].axis('equal')

    # 配准后 - XZ视图
    axes[1, 1].scatter(aligned_pts[:, 0], aligned_pts[:, 2], s=0.1, c='orange', alpha=0.5)
    axes[1, 1].scatter(target_pts[:, 0], target_pts[:, 2], s=0.1, c='blue', alpha=0.5)
    axes[1, 1].set_title('After Registration (XZ)')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Z')
    axes[1, 1].axis('equal')

    # 配准后 - YZ视图
    axes[1, 2].scatter(aligned_pts[:, 1], aligned_pts[:, 2], s=0.1, c='orange', alpha=0.5)
    axes[1, 2].scatter(target_pts[:, 1], target_pts[:, 2], s=0.1, c='blue', alpha=0.5)
    axes[1, 2].set_title('After Registration (YZ)')
    axes[1, 2].set_xlabel('Y')
    axes[1, 2].set_ylabel('Z')
    axes[1, 2].axis('equal')

    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"已保存: {output_image}")
    print(f"  源点云: {len(source_pts)} 点")
    print(f"  目标点云: {len(target_pts)} 点")

def main():
    data_dir = "/home/user/PKSS-Align/PKSS_Align_EXE/Data"

    # 飞机数据
    print("生成飞机点云配准可视化...")
    visualize_registration(
        os.path.join(data_dir, "airplane_0001_s.ply"),
        os.path.join(data_dir, "airplane_0001_t.ply"),
        os.path.join(data_dir, "airplane_0001_s_aligned_open3d.ply"),
        "/home/user/PKSS-Align/airplane_registration_result.png",
        "Airplane Point Cloud Registration (ModelNet40)"
    )

    # 室内场景数据
    print("\n生成室内场景点云配准可视化...")
    visualize_registration(
        os.path.join(data_dir, "01f_s.ply"),
        os.path.join(data_dir, "01f_t.ply"),
        os.path.join(data_dir, "01f_s_aligned_open3d.ply"),
        "/home/user/PKSS-Align/indoor_registration_result.png",
        "Indoor Scene Point Cloud Registration (S3DIS)"
    )

if __name__ == "__main__":
    main()
