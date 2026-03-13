import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')  # TODO

import argparse
import cv2
import glob
import imageio
import logging
import os
import numpy as np
from typing import List
import copy

import omegaconf
import open3d as o3d
import torch
import yaml
import time
import tensorrt as trt

import sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from Utils import vis_disparity, depth2xyzmap, toOpen3dCloud, set_seed
from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


'''
python scripts/run_demo_tensorrt.py \
        --left_img assets/left.png \
        --right_img assets/right.png \
        --save_path output \
        --pretrained pretrained_models/foundation_stereo.plan \
        --height 512 \
        --width 512 \
        --pc \
        --z_far 100.0
'''

def preprocess(image_path, args):
    input_image = imageio.imread(image_path)
    if args.height and args.width:
      input_image = cv2.resize(input_image, (args.width, args.height))
    resized_image = torch.as_tensor(input_image.copy()).float()[None].permute(0,3,1,2).contiguous()
    return resized_image, input_image


def get_onnx_model(args):
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise ImportError(
            "Failed to import onnxruntime. If you only use TensorRT `.plan/.engine`, "
            "please run with --pretrained pointing to that file. "
            "For ONNX mode, install a NumPy-compatible onnxruntime build."
        ) from exc
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = ort.InferenceSession(args.pretrained, sess_options=session_options, providers=['CUDAExecutionProvider'])
    return model


def get_engine_model(args):
    with open(args.pretrained, 'rb') as file:
        engine_data = file.read()
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {args.pretrained}")
    return engine


def get_trt_static_hw(engine):
    """Get static (H, W) from TRT engine input tensor if available."""
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        shape = tuple(engine.get_tensor_shape(name))
        if len(shape) == 4 and shape[2] > 0 and shape[3] > 0:
            return int(shape[2]), int(shape[3])
    return None


def run_trt_engine(engine, left_np, right_np):
    """Run TensorRT engine and return numpy outputs."""
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context.")

    input_names, output_names = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)

    if len(input_names) != 2:
        raise RuntimeError(f"Expected 2 inputs, got {len(input_names)}: {input_names}")

    left_name, right_name = input_names[0], input_names[1]
    left_t = torch.from_numpy(left_np).cuda()
    right_t = torch.from_numpy(right_np).cuda()

    if not context.set_input_shape(left_name, tuple(left_t.shape)):
        raise ValueError(f"Failed to set TRT input shape for {left_name}: {tuple(left_t.shape)}")
    if not context.set_input_shape(right_name, tuple(right_t.shape)):
        raise ValueError(f"Failed to set TRT input shape for {right_name}: {tuple(right_t.shape)}")

    context.set_tensor_address(left_name, int(left_t.data_ptr()))
    context.set_tensor_address(right_name, int(right_t.data_ptr()))

    dtype_map = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }
    output_tensors = []
    for name in output_names:
        out_shape = tuple(context.get_tensor_shape(name))
        out_dtype = dtype_map.get(engine.get_tensor_dtype(name), torch.float32)
        out_t = torch.empty(out_shape, dtype=out_dtype, device='cuda')
        context.set_tensor_address(name, int(out_t.data_ptr()))
        output_tensors.append(out_t)

    if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
        raise RuntimeError("TensorRT execute_async_v3 failed.")
    torch.cuda.current_stream().synchronize()

    return [t.detach().cpu().numpy() for t in output_tensors]


def inference(left_img_path: str, right_img_path: str, model, args: argparse.Namespace):
    if args.pretrained.endswith('.engine') or args.pretrained.endswith('.plan'):
        static_hw = get_trt_static_hw(model)
        if static_hw is not None:
            h, w = static_hw
            if args.height != h or args.width != w:
                logging.warning(
                    "TRT engine expects %dx%d, overriding requested %dx%d.",
                    h, w, args.height, args.width
                )
                args.height, args.width = h, w

    left_img, input_left = preprocess(left_img_path, args)
    right_img, _ = preprocess(right_img_path, args)

    for _ in range(10):
      torch.cuda.synchronize()
      start_time = time.time()
      if args.pretrained.endswith('.onnx'):
          outputs = model.run(None, {'left': left_img.numpy(), 'right': right_img.numpy()})
      else:
          outputs = run_trt_engine(model, left_img.numpy(), right_img.numpy())
      left_disp = outputs[0]
      left_conf = outputs[1] if len(outputs) > 1 else None
      torch.cuda.synchronize()
      end_time = time.time()
      logging.info(f'Inference time: {end_time - start_time:.3f} seconds')

    left_disp = left_disp.squeeze()  # HxW
    left_conf = left_conf.squeeze() if left_conf is not None else None

    vis = vis_disparity(left_disp)
    if vis.shape[:2] != input_left.shape[:2]:
        vis = cv2.resize(vis, (input_left.shape[1], input_left.shape[0]), interpolation=cv2.INTER_NEAREST)
    vis = np.concatenate([input_left, vis], axis=1)
    imageio.imwrite(os.path.join(args.save_path, 'visual', left_img_path.split('/')[-1]), vis)
    np.save(os.path.join(args.save_path, 'continuous/disparity', left_img_path.split('/')[-1].split('.')[0] + '.npy'), left_disp.astype(np.float32))
    if left_conf is not None:
        conf_vis = np.clip(left_conf * 255.0, 0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(args.save_path, 'continuous/confidence', left_img_path.split('/')[-1]), conf_vis)
        np.save(os.path.join(args.save_path, 'continuous/confidence', left_img_path.split('/')[-1].split('.')[0] + '.npy'), left_conf.astype(np.float32))

    if args.pc:
        save_path = left_img_path.split('/')[-1].split('.')[0] + '.ply'
        baseline = 193.001/1e3
        doffs = 0
        K = np.array([1998.842, 0, 588.364,
                    0, 1998.842, 505.864,
                    0,0,1]).reshape(3,3)
        depth = K[0,0]*baseline/(left_disp + doffs)
        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1,3), input_left.reshape(-1,3))
        keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)
        o3d.io.write_point_cloud(os.path.join(args.save_path, 'cloud', save_path), pcd)



def parse_args() -> omegaconf.OmegaConf:
    parser = argparse.ArgumentParser(description='Stereo 2025')
    code_dir = os.path.dirname(os.path.realpath(__file__))

    # File options
    parser.add_argument('--left_img', '-l', required=True, help='Path to left image.')
    parser.add_argument('--right_img', '-r', required=True, help='Path to right image.')
    parser.add_argument('--save_path', '-s', default=f'{code_dir}/../output', help='Path to save results.')
    parser.add_argument('--pretrained', default='2024-12-13-23-51-11/model_best_bp2.pth', help='Path to pretrained model')

    # Inference options
    parser.add_argument('--height', type=int, default=448, help='Image height')
    parser.add_argument('--width', type=int, default=672, help='Image width')
    parser.add_argument('--pc', action='store_true', help='Save point cloud')
    parser.add_argument('--z_far', default=100, type=float, help='max depth to clip in point cloud')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    paths = ['continuous/disparity', 'visual', 'denoised_cloud', 'cloud']
    paths.append('continuous/confidence')
    for p in paths:
        os.makedirs(os.path.join(args.save_path, p), exist_ok=True)

    assert os.path.isfile(args.pretrained), f'Pretrained model {args.pretrained} not found'
    logging.info('Pretrained model loaded from %s', args.pretrained)
    set_seed(0)
    if args.pretrained.endswith('.onnx'):
        model = get_onnx_model(args)
    elif args.pretrained.endswith('.engine') or args.pretrained.endswith('.plan'):
        model = get_engine_model(args)
    else:
        assert False, f'Unknown model format {args.pretrained}.'

    inference(args.left_img, args.right_img, model, args)

if __name__ == '__main__':
    main()