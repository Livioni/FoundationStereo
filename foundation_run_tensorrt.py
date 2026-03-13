#!/usr/bin/env python3
"""
TensorRT version of foundation_run.py.
Keeps original processing flow unchanged and only swaps inference backend.
"""

import argparse
import os

import numpy as np
import torch
import tensorrt as trt

import foundation_run as base


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTStereoModel:
    def __init__(self, engine_path: str):
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"无法反序列化 TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("创建 TensorRT execution context 失败")

        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if len(self.input_names) != 2:
            raise RuntimeError(
                f"TensorRT engine 输入数量异常，期望2个，实际{len(self.input_names)}: {self.input_names}"
            )

        self.dtype_map = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }

    def infer(self, left_np: np.ndarray, right_np: np.ndarray):
        left_name, right_name = self.input_names[0], self.input_names[1]

        left_t = torch.from_numpy(left_np).to("cuda")
        right_t = torch.from_numpy(right_np).to("cuda")

        if not self.context.set_input_shape(left_name, tuple(left_t.shape)):
            raise RuntimeError(f"设置输入形状失败: {left_name} -> {tuple(left_t.shape)}")
        if not self.context.set_input_shape(right_name, tuple(right_t.shape)):
            raise RuntimeError(f"设置输入形状失败: {right_name} -> {tuple(right_t.shape)}")

        self.context.set_tensor_address(left_name, int(left_t.data_ptr()))
        self.context.set_tensor_address(right_name, int(right_t.data_ptr()))

        output_tensors = []
        for name in self.output_names:
            out_shape = tuple(self.context.get_tensor_shape(name))
            out_dtype = self.dtype_map.get(self.engine.get_tensor_dtype(name), torch.float32)
            out_t = torch.empty(out_shape, dtype=out_dtype, device="cuda")
            self.context.set_tensor_address(name, int(out_t.data_ptr()))
            output_tensors.append(out_t)

        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT 推理执行失败 (execute_async_v3)")
        torch.cuda.current_stream().synchronize()

        return [t.detach().cpu().numpy() for t in output_tensors]


def load_model(args):
    if not os.path.exists(args.ckpt_dir):
        raise FileNotFoundError(f"TensorRT engine 不存在: {args.ckpt_dir}")
    if not (args.ckpt_dir.endswith(".engine") or args.ckpt_dir.endswith(".plan")):
        raise ValueError(
            f"--ckpt_dir 需要 TensorRT engine 文件(.engine/.plan)，当前为: {args.ckpt_dir}"
        )
    model = TensorRTStereoModel(args.ckpt_dir)
    return model, args


def infer_disparity(model, args, img0, img1):
    scale = args.scale
    if scale <= 0 or scale > 1:
        raise ValueError("scale must be <=1 and >0")

    if scale != 1.0:
        img0 = base.cv2.resize(img0, fx=scale, fy=scale, dsize=None)
        img1 = base.cv2.resize(img1, fx=scale, fy=scale, dsize=None)

    h, w = img0.shape[:2]
    img0_t = torch.as_tensor(img0).to(args.device).float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1).to(args.device).float()[None].permute(0, 3, 1, 2)
    padder = base.InputPadder(img0_t.shape, divis_by=32, force_square=False)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    outputs = model.infer(
        img0_t.detach().cpu().numpy().astype(np.float32),
        img1_t.detach().cpu().numpy().astype(np.float32),
    )

    disp = torch.from_numpy(outputs[0]).to(args.device).float()
    disp = padder.unpad(disp)
    disp = disp.data.cpu().numpy().reshape(h, w)

    if len(outputs) > 1:
        conf = torch.from_numpy(outputs[1]).to(args.device).float()
        conf = padder.unpad(conf)
        conf = conf.data.cpu().numpy().reshape(h, w)
    else:
        # Keep downstream behavior unchanged: always allow saving conf mask.
        conf = np.ones((h, w), dtype=np.float32)

    return disp, conf


def _patch_parser_for_trt(parser: argparse.ArgumentParser):
    parser.set_defaults(ckpt_dir="pretrained_models/foundation_stereo.plan")
    for action in parser._actions:
        if "--ckpt_dir" in action.option_strings:
            action.help = "TensorRT engine 路径 (.plan/.engine)"
        elif "--save_visualization" in action.option_strings:
            action.help = "保存深度图可视化（TRT版本同样支持）"
        elif "--remove_invisible" in action.option_strings:
            # Support both styles:
            #   --remove_invisible        -> 1
            #   --remove_invisible 0/1    -> explicit value
            action.nargs = "?"
            action.const = 1
            action.type = int
            action.help = "移除左右视角不重叠区域（TRT版本同样支持）"

    # Keep base defaults for intrinsics (fx/fy/cx/cy), and only baseline is optional.
    # This allows "--baseline xxx" to work directly without requiring HDF5.
    parser.set_defaults(baseline=None)


def main(args):
    base.set_logging()
    torch.autograd.set_grad_enabled(False)

    if not args.video_folder:
        raise ValueError("必须指定 --video_folder 参数")
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境不可用 CUDA，无法运行 TensorRT 推理")

    args.device = args.device
    base.logging.info("使用设备: %s", args.device)

    model, args = load_model(args)

    # Keep original process flow; only replace base infer function with TRT backend.
    base.infer_disparity = infer_disparity
    base.process_video_folder(args.video_folder, args, model)


if __name__ == "__main__":
    parser = base.get_args_parser()
    _patch_parser_for_trt(parser)
    args = parser.parse_args()
    main(args)
