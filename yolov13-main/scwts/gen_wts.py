# gen_wts.py
import torch
import argparse
import os

def export_wts_from_ckpt(ckpt_file, output_file):
    # 加载 checkpoint
    ckpt = torch.load(ckpt_file, map_location='cpu')
    
    # 获取 state_dict
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            # 尝试直接使用
            state_dict = ckpt
    else:
        state_dict = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt

    max_vals = 32768
    with open(output_file, 'w') as f:
        f.write(f"{len(state_dict.keys())}\n")
        for k, v in state_dict.items():
            if k.endswith('num_batches_tracked'):
                continue
            vr = v.reshape(-1).cpu().numpy()
            rows = len(vr)
            f.write(f"{k} {rows}")
            for i, value in enumerate(vr):
                if i % max_vals == 0 and i > 0:
                    f.write("\n")
                f.write(f" {value:.8f}")
            f.write("\n")
    print(f"✅ 权重已导出到: {output_file}")
    print(f"🔍 总共导出 {len(state_dict.keys())} 个参数层")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='权重文件')
    parser.add_argument('--output', type=str, default='yolov13.wts', help='输出 .wts 文件')
    args = parser.parse_args()

    print(f"📂 正在加载权重: {args.weights}")
    if not os.path.exists(args.weights):
        print(f"❌ 文件不存在: {args.weights}")
        return

    export_wts_from_ckpt(args.weights, args.output)

if __name__ == "__main__":
    main()
