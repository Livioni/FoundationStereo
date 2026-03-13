import warnings, argparse, logging, os, sys
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')
import omegaconf, yaml, torch,pdb
from omegaconf import OmegaConf
from core.foundation_stereo import FoundationStereo


class FoundationStereoOnnx(FoundationStereo):
    def __init__(self, args):
        super().__init__(args)

    @torch.no_grad()
    def forward(self, left, right):
        """Export disparity and confidence outputs for ONNX/TRT."""
        with torch.amp.autocast('cuda', enabled=True):
            out = FoundationStereo.forward(self, left, right, iters=self.args.valid_iters, test_mode=True)
            if isinstance(out, tuple):
                disp, conf = out
            else:
                disp = out
                conf = torch.ones_like(disp)
        return disp, conf



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default=f'{code_dir}/../output/foundation_stereo.onnx', help='Path to save results.')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2-001.pth', type=str, help='pretrained model path')
    parser.add_argument('--height', type=int, default=448)
    parser.add_argument('--width', type=int, default=672)
    parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    torch.autograd.set_grad_enabled(False)

    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    for k in args.__dict__:
      cfg[k] = args.__dict__[k]
    if 'vit_size' not in cfg:
      cfg['vit_size'] = 'vitl'
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")
    model = FoundationStereoOnnx(cfg)
    ckpt = torch.load(ckpt_dir, weights_only=False)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()


    left_img = torch.randn(1, 3, args.height, args.width).cuda().float()
    right_img = torch.randn(1, 3, args.height, args.width).cuda().float()

    torch.onnx.export(
        model,
        (left_img, right_img),
        args.save_path,
        opset_version=16,
        input_names = ['left', 'right'],
        output_names = ['disp', 'conf'],
        dynamic_axes={
            'left': {0 : 'batch_size'},
            'right': {0 : 'batch_size'},
            'disp': {0 : 'batch_size'},
            'conf': {0 : 'batch_size'}
        },
    )

