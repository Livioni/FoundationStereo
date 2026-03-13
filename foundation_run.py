#!/usr/bin/env python3
"""
Batch process stereo image pairs using FoundationStereo model to generate depth maps.

支持两种处理模式：

1. 传统批量模式 (兼容 demo/batch_stereo_depth_pytorch.py):
   - Input: dataset root with camera folders
   - Output: save left-view depth to each camera's depth_npy folder

2. 视频文件夹模式 (新增):
   - Input: folder containing stereo video files (stereo_left.mp4, stereo_right.mp4)
            and annotation.hdf5 with camera intrinsics/baseline
   - Processing: extract video frames to image/left and image/right folders
   - Output: save left-view depth to depth folder as .npy files (frame_xxxx_left.npy)
   - 可选择手动指定相机参数 (fx, fy, cx, cy, baseline)，此时不需要HDF5文件

使用示例:
   # 视频文件夹模式（从HDF5读取参数）
   python foundationsstereos_run.py --video_folder datasets/ropedia/ep3

   # 手动指定相机参数（不需要HDF5文件）
   python foundationsstereos_run.py --video_folder datasets/my_data \
       --fx 800.0 --fy 800.0 --cx 640.0 --cy 480.0 --baseline 0.1
"""

import argparse
import logging
import os
import sys
import glob
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
import torch
import imageio.v2 as imageio
from tqdm import tqdm
from omegaconf import OmegaConf
import h5py

# Add project root and FoundationStereo to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
foundation_root = os.path.join(project_root, "FoundationStereo")
sys.path.insert(0, project_root)
sys.path.insert(0, foundation_root)

from FoundationStereo.core.utils.utils import InputPadder
from FoundationStereo.core.foundation_stereo import FoundationStereo

def extract_intrinsic_matrix_from_hdf5(hdf5_file_path):
    """
    从HDF5文件中提取内参矩阵

    Args:
        hdf5_file_path: HDF5文件路径

    Returns:
        np.array: 3x3内参矩阵K
    """
    try:
        with h5py.File(hdf5_file_path, 'r') as f:
            # 读取 calibration/cam01/K
            k_params = f['calibration/cam01/K'][:]

        if len(k_params) != 4:
            print(f"错误: 期望4个内参参数，得到 {len(k_params)} 个")
            return None

        # 解析内参参数
        fx, fy, cx, cy = k_params

        # 构建3x3内参矩阵
        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float32)

        return K

    except KeyError as e:
        print(f"错误: 在HDF5文件中找不到键 {e}")
        return None
    except Exception as e:
        print(f"提取内参矩阵失败: {e}")
        return None


def extract_baseline_from_hdf5(hdf5_file_path):
    """
    从HDF5文件中提取baseline

    Args:
        hdf5_file_path: HDF5文件路径

    Returns:
        float: baseline值（米）
    """
    try:
        with h5py.File(hdf5_file_path, 'r') as f:
            # 读取 calibration/cam01/baseline
            baseline = f['calibration/cam01/baseline'][()]

        return float(baseline)

    except KeyError as e:
        print(f"错误: 在HDF5文件中找不到键 {e}")
        return None
    except Exception as e:
        print(f"提取baseline失败: {e}")
        return None


def extract_video_frames(video_path, output_dir, frame_prefix="frame_", extension="png"):
    """
    从视频文件中提取帧并保存为图像

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        frame_prefix: 帧文件名前缀
        extension: 图像文件扩展名

    Returns:
        int: 提取的帧数
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    logging.info(f"开始从 {video_path} 提取帧到 {output_dir}")

    with tqdm(total=frame_count, desc=f"提取 {os.path.basename(video_path)}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 转换为RGB格式（OpenCV默认是BGR）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 生成文件名，格式为 frame_xxxx_rgb.png
            frame_filename = f"frame_{frame_idx:05d}_rgb.png"

            output_path = os.path.join(output_dir, frame_filename)
            imageio.imsave(output_path, frame_rgb)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    logging.info(f"完成提取 {frame_idx} 帧到 {output_dir}")
    return frame_idx
    

def set_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="批量处理立体图像对生成深度图 (FoundationStereo)"
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        help="包含立体视频文件的文件夹路径（例如 datasets/ropedia/ep3）",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="pretrained_models/23-51-11/model_best_bp2-001.pth",
        help="预训练模型路径",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备")
    parser.add_argument("--scale", default=1.0, type=float, help="图像缩放比例(<=1)")
    parser.add_argument(
        "--hiera",
        default=False,
        type=bool,
        help="hierarchical inference (适用于>1K分辨率)", 
    )
    parser.add_argument(
        "--valid_iters", type=int, default=64, help="前向迭代次数"
    )
    parser.add_argument(
        "--image_extension",
        type=str,
        default="png",
        help="图像文件扩展名 (png, jpg, etc.)",
    )
    parser.add_argument(
        "--save_visualization",
        action="store_true",
        help="保存深度图可视化",
    )
    parser.add_argument(
        "--remove_invisible",
        type=int,
        default=True,
        help="移除左右视角不重叠区域（与run_demo.py一致）",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=None,
        help="置信度阈值，低于该阈值的像素深度将置0（0-1，默认不启用）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="兼容模式输出目录（单相机处理）",
    )
    parser.add_argument(
        "--crop_height",
        type=int,
        default=None,
        help="从中心裁剪的高度（未填则不裁剪）",
    )
    parser.add_argument(
        "--crop_width",
        type=int,
        default=None,
        help="从中心裁剪的宽度（未填则不裁剪）",
    )
    parser.add_argument(
        "--fx",
        type=float,
        default=200,
        help="相机内参 fx（如果指定，将不从HDF5读取）",
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=200,
        help="相机内参 fy（如果指定，将不从HDF5读取）",
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=256,
        help="相机内参 cx（如果指定，将不从HDF5读取）",
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=256,
        help="相机内参 cy（如果指定，将不从HDF5读取）",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="相机基线 baseline（如果指定，将不从HDF5读取）",
    )

    return parser


def load_camera_matrix(intrinsic_path):
    intrinsic_data = np.load(intrinsic_path, allow_pickle=True)
    if isinstance(intrinsic_data, np.ndarray) and intrinsic_data.dtype == object:
        intrinsic_data = intrinsic_data.item()
        if isinstance(intrinsic_data, dict):
            for key in ["K", "intrinsic", "camera_matrix", "intrinsics"]:
                if key in intrinsic_data:
                    intrinsic_matrix = intrinsic_data[key]
                    break
            else:
                raise ValueError(
                    f"无法从字典中找到内参矩阵，可用的键: {intrinsic_data.keys()}"
                )
        else:
            intrinsic_matrix = intrinsic_data
    else:
        intrinsic_matrix = intrinsic_data

    intrinsic_matrix = np.array(intrinsic_matrix).reshape(3, 3)
    fx = intrinsic_matrix[0, 0]
    return intrinsic_matrix, fx


def load_extrinsics_array(extrinsic_path):
    extrinsics = np.load(extrinsic_path, allow_pickle=True)
    return extrinsics


def compute_baseline_per_frame(extrinsic_left, extrinsic_right):
    if extrinsic_left.shape[0] == 3:
        t_left = extrinsic_left[:, 3]
    else:
        t_left = extrinsic_left[:3, 3]

    if extrinsic_right.shape[0] == 3:
        t_right = extrinsic_right[:, 3]
    else:
        t_right = extrinsic_right[:3, 3]

    baseline_m = float(np.linalg.norm(t_right - t_left))
    return baseline_m


def get_image_pairs(left_folder, right_folder, extension="png"):
    def extract_frame_number(filepath):
        """从文件名中提取帧号，例如 'frame_00293_rgb.png' -> 293"""
        basename = os.path.splitext(os.path.basename(filepath))[0]
        # 使用正则表达式匹配 frame_XXXXX_rgb 格式
        match = re.search(r'frame_(\d+)_rgb', basename)
        if match:
            return int(match.group(1))
        else:
            # 如果不匹配，尝试其他格式或返回0
            return 0

    left_images = sorted(
        glob.glob(os.path.join(left_folder, f"*.{extension}")),
        key=extract_frame_number,
    )
    # left_images = sorted(left_images)
    if len(left_images) == 0:
        raise ValueError(f"在 {left_folder} 中未找到 .{extension} 图像")

    image_pairs = []
    for left_path in left_images:
        filename = os.path.basename(left_path)
        right_path = os.path.join(right_folder, filename)
        if os.path.exists(right_path):
            image_pairs.append((left_path, right_path, filename))
        else:
            logging.warning("未找到对应的右图像: %s", right_path)

    logging.info("找到 %d 对立体图像", len(image_pairs))
    return image_pairs


def save_depth_visualization(depth_map, output_path):
    depth_vis = depth_map.copy()
    valid_mask = depth_vis > 0

    if np.any(valid_mask):
        depth_vis_clipped = np.clip(depth_vis, 0, 10)
        depth_vis_log = np.log(depth_vis_clipped + 1)
        min_val = np.min(depth_vis_log[valid_mask])
        max_val = np.max(depth_vis_log[valid_mask])
        depth_vis_norm = np.zeros_like(depth_vis_log, dtype=np.uint8)
        if max_val > min_val:
            depth_vis_norm[valid_mask] = (
                (depth_vis_log[valid_mask] - min_val) / (max_val - min_val)
            ) * 255
        depth_vis = depth_vis_norm.astype(np.uint8)
    else:
        depth_vis = np.zeros_like(depth_vis, dtype=np.uint8)

    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, depth_colored)


def load_model(args):
    if not os.path.exists(args.ckpt_dir):
        raise FileNotFoundError(f"模型权重不存在: {args.ckpt_dir}")

    cfg = OmegaConf.load(f"{os.path.dirname(args.ckpt_dir)}/cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)

    model = FoundationStereo(args)
    ckpt = torch.load(args.ckpt_dir, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return model, args


def infer_disparity(model, args, img0, img1):
    scale = args.scale
    if scale <= 0 or scale > 1:
        raise ValueError("scale must be <=1 and >0")

    if scale != 1.0:
        img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)

    h, w = img0.shape[:2]
    img0_t = torch.as_tensor(img0).to(args.device).float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1).to(args.device).float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    with torch.no_grad(), torch.amp.autocast('cuda', enabled="cuda" in str(args.device)):
        if not args.hiera:
            out = model.forward(img0_t, img1_t, iters=args.valid_iters, test_mode=True)
        else:
            out = model.run_hierachical(
                img0_t, img1_t, iters=args.valid_iters, test_mode=True, small_ratio=0.5
            )
    if isinstance(out, tuple):
        disp, conf = out
        conf = padder.unpad(conf.float())
        conf = conf.data.cpu().numpy().reshape(h, w)
    else:
        disp, conf = out, None
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(h, w)
    return disp, conf


def central_crop_image(image, target_height, target_width):
    """
    以图像中心为基准裁剪到指定分辨率。

    Args:
        image: 原始图像（H x W x C）。
        target_height: 目标高度。
        target_width: 目标宽度。

    Returns:
        裁剪后的图像。
    """
    h, w = image.shape[:2]
    if target_height is None or target_width is None:
        return image
    if target_height > h or target_width > w:
        raise ValueError(
            f"裁剪尺寸 ({target_height},{target_width}) 超过原图 ({h},{w})"
        )

    start_y = (h - target_height) // 2
    start_x = (w - target_width) // 2
    return image[start_y : start_y + target_height, start_x : start_x + target_width]


def process_video_folder(video_folder, args, model):
    """
    处理包含立体视频的文件夹：
    1. 提取视频帧到图像文件夹
    2. 从HDF5读取内参和baseline
    3. 生成深度图

    Args:
        video_folder: 包含视频文件的文件夹路径
        args: 参数对象
        model: FoundationStereo模型
    """
    video_folder = Path(video_folder)
    if not video_folder.exists():
        raise ValueError(f"视频文件夹不存在: {video_folder}")

    # 检查视频文件
    left_video = video_folder / "stereo_left.mp4"
    right_video = video_folder / "stereo_right.mp4"
    hdf5_file = video_folder / "annotation.hdf5"

    if not left_video.exists():
        raise ValueError(f"找不到左眼视频文件: {left_video}")
    if not right_video.exists():
        raise ValueError(f"找不到右眼视频文件: {right_video}")

    # 检查是否提供了相机参数，或者HDF5文件是否存在
    has_manual_params = (args.fx is not None and args.fy is not None and
                        args.cx is not None and args.cy is not None and args.baseline is not None)
    if not has_manual_params and not hdf5_file.exists():
        raise ValueError(f"找不到HDF5标注文件: {hdf5_file}，且未提供完整的相机参数")

    # 创建输出文件夹
    images_left_dir = video_folder / "images" / "left"
    images_right_dir = video_folder / "images" / "right"
    depth_dir = video_folder / "depths"

    logging.info(f"处理视频文件夹: {video_folder}")

    image_pattern = f"*.{args.image_extension}"
    left_has_images = images_left_dir.exists() and any(
        images_left_dir.glob(image_pattern)
    )
    right_has_images = images_right_dir.exists() and any(
        images_right_dir.glob(image_pattern)
    )
    left_frame_count = right_frame_count = 0

    if left_has_images and right_has_images:
        left_frame_count = sum(1 for _ in images_left_dir.glob(image_pattern))
        right_frame_count = sum(1 for _ in images_right_dir.glob(image_pattern))
        logging.info(
            "检测到已有左右眼图像，跳过视频帧提取（%d vs %d）",
            left_frame_count,
            right_frame_count,
        )
    else:
        logging.info("并行提取左右眼视频帧...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_left = executor.submit(
                extract_video_frames, str(left_video), str(images_left_dir)
            )
            future_right = executor.submit(
                extract_video_frames, str(right_video), str(images_right_dir)
            )
            left_frame_count = future_left.result()
            right_frame_count = future_right.result()
        logging.info("左右眼视频帧提取完成")

    if left_frame_count != right_frame_count:
        logging.warning(
            f"左右视频帧数不匹配: 左={left_frame_count}, 右={right_frame_count}"
        )

    # 读取内参和baseline
    if args.fx is not None and args.fy is not None and args.cx is not None and args.cy is not None and args.baseline is not None:
        # 使用指定的相机参数
        logging.info("使用指定的相机参数: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f, baseline=%.4f",
                    args.fx, args.fy, args.cx, args.cy, args.baseline)
        intrinsic_matrix = np.array([
            [args.fx,  0, args.cx],
            [ 0, args.fy, args.cy],
            [ 0,  0,  1]
        ], dtype=np.float32)
        baseline = args.baseline
    else:
        # 从HDF5文件读取相机参数
        logging.info("从HDF5文件读取相机参数...")
        intrinsic_matrix = extract_intrinsic_matrix_from_hdf5(str(hdf5_file))
        baseline = extract_baseline_from_hdf5(str(hdf5_file))

        if intrinsic_matrix is None or baseline is None:
            raise ValueError("无法从HDF5文件读取相机参数，且未指定相机参数")


    # 获取图像对
    image_pairs = get_image_pairs(str(images_left_dir), str(images_right_dir), args.image_extension)
    if len(image_pairs) == 0:
        logging.error("未找到图像对")
        return

    # 处理深度图生成
    depth_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = None
    if args.save_visualization:
        visualization_dir = video_folder / "visualization"
        visualization_dir.mkdir(parents=True, exist_ok=True)
    conf_mask_dir = video_folder / "conf_mask"
    conf_mask_dir.mkdir(parents=True, exist_ok=True)

    fx = intrinsic_matrix[0, 0] * args.scale

    for frame_idx, (left_path, right_path, filename) in enumerate(
        tqdm(image_pairs, desc="生成深度图")
    ):
        try:
            img0 = imageio.imread(left_path)
            img1 = imageio.imread(right_path)
            if img0.ndim == 2:
                img0 = np.stack([img0] * 3, axis=-1)
            if img1.ndim == 2:
                img1 = np.stack([img1] * 3, axis=-1)
            if img0.shape[-1] == 4:
                img0 = img0[:, :, :3]
            if img1.shape[-1] == 4:
                img1 = img1[:, :, :3]
            crop_height = args.crop_height
            crop_width = args.crop_width
            if bool(crop_height is not None) ^ bool(crop_width is not None):
                raise ValueError("需要同时指定 --crop_height 和 --crop_width 才能裁剪图像")
            if crop_height is not None and crop_width is not None:
                img0 = central_crop_image(img0, crop_height, crop_width)
                img1 = central_crop_image(img1, crop_height, crop_width)
                logging.debug(
                    "裁剪到 %dx%d 中心区域再送入模型", crop_height, crop_width
                )

            disp, conf = infer_disparity(model, args, img0, img1)
            if args.remove_invisible:
                yy, xx = np.meshgrid(
                    np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing="ij"
                )
                us_right = xx - disp
                disp[us_right < 0] = np.inf

            disp_safe = disp.copy()
            disp_safe[disp_safe <= 0] = np.nan
            depth = fx * baseline / disp_safe
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            # 保存为 .npy 格式
            base_name = os.path.splitext(filename)[0]
            # depth_path = depth_dir / f"{base_name}.npy"
            # np.save(str(depth_path), depth)
            depth_mm = np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)
            depth_mm_path = depth_dir / f"{base_name}.png"
            cv2.imwrite(str(depth_mm_path), depth_mm)
            if args.save_visualization:
                vis_path = visualization_dir / f"{base_name}.png"
                save_depth_visualization(depth, str(vis_path))
            if conf is not None:
                conf_uint16 = np.clip(np.round(conf * 65535.0), 0, 65535).astype(
                    np.uint16
                )
                conf_mask_path = conf_mask_dir / f"{base_name}.png"
                cv2.imwrite(str(conf_mask_path), conf_uint16)

        except Exception as exc:
            logging.exception(f"处理帧 {filename} (frame {frame_idx}) 出错: {str(exc)}")
            continue

    logging.info(f"深度图生成完成，输出目录: {depth_dir}")


def main(args):
    set_logging()
    torch.autograd.set_grad_enabled(False)

    if not args.video_folder:
        raise ValueError("必须指定 --video_folder 参数")

    # 视频文件夹处理模式
    device = args.device if torch.cuda.is_available() else "cpu"
    args.device = device
    logging.info("使用设备: %s", device)

    model, args = load_model(args)
    model = model.to(args.device)
    model.eval()

    process_video_folder(args.video_folder, args, model)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
