import argparse
import re
import torch
from tqdm import tqdm
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


WEIGHT_MAPPING = {
    "blocks\.(\d+)\.ln0\.weight": "model.layers.{}.pre_norm.weight".format,
    "blocks\.(\d+)\.ln0\.bias": "model.layers.{}.pre_norm.bias".format,
    "blocks\.(\d+)\.ln1\.weight": "model.layers.{}.attn_norm.weight".format,
    "blocks\.(\d+)\.ln1\.bias": "model.layers.{}.attn_norm.bias".format,
    "blocks\.(\d+)\.ln2\.weight": "model.layers.{}.ffn_norm.weight".format,
    "blocks\.(\d+)\.ln2\.bias": "model.layers.{}.ffn_norm.bias".format,
    "blocks\.(\d+)\.att\.x_r": "model.layers.{}.attn.x_x".format,
    "blocks\.(\d+)\.att\.x_w": "model.layers.{}.attn.x_x".format,
    "blocks\.(\d+)\.att\.x_k": "model.layers.{}.attn.x_x".format,
    "blocks\.(\d+)\.att\.x_v": "model.layers.{}.attn.x_x".format,
    "blocks\.(\d+)\.att\.x_a": "model.layers.{}.attn.x_x".format,
    "blocks\.(\d+)\.att\.x_g": "model.layers.{}.attn.x_x".format,
    "blocks\.(\d+)\.att\.v0": "model.layers.{}.attn.v_lora.lora.2.bias".format,
    "blocks\.(\d+)\.att\.v1": "model.layers.{}.attn.v_lora.lora.0.weight".format,
    "blocks\.(\d+)\.att\.v2": "model.layers.{}.attn.v_lora.lora.2.weight".format,
    "blocks\.(\d+)\.att\.w1": "model.layers.{}.attn.w_lora.lora.0.weight".format,
    "blocks\.(\d+)\.att\.w2": "model.layers.{}.attn.w_lora.lora.2.weight".format,
    "blocks\.(\d+)\.att\.w0": "model.layers.{}.attn.w_lora.lora.2.bias".format,
    "blocks\.(\d+)\.att\.a1": "model.layers.{}.attn.a_lora.lora.0.weight".format,
    "blocks\.(\d+)\.att\.a2": "model.layers.{}.attn.a_lora.lora.2.weight".format,
    "blocks\.(\d+)\.att\.a0": "model.layers.{}.attn.a_lora.lora.2.bias".format,
    "blocks\.(\d+)\.att\.g1": "model.layers.{}.attn.g_lora.lora.0.weight".format,
    "blocks\.(\d+)\.att\.g2": "model.layers.{}.attn.g_lora.lora.2.weight".format,
    "blocks\.(\d+)\.att\.k_k": "model.layers.{}.attn.k_k".format,
    "blocks\.(\d+)\.att\.k_a": "model.layers.{}.attn.k_a".format,
    "blocks\.(\d+)\.att\.r_k": "model.layers.{}.attn.r_k".format,
    "blocks\.(\d+)\.att\.receptance\.weight": "model.layers.{}.attn.r_proj.weight".format,
    "blocks\.(\d+)\.att\.key\.weight": "model.layers.{}.attn.k_proj.weight".format,
    "blocks\.(\d+)\.att\.value\.weight": "model.layers.{}.attn.v_proj.weight".format,
    "blocks\.(\d+)\.att\.output\.weight": "model.layers.{}.attn.o_proj.weight".format,
    "blocks\.(\d+)\.att\.ln_x\.weight": "model.layers.{}.attn.g_norm.weight".format,
    "blocks\.(\d+)\.att\.ln_x\.bias": "model.layers.{}.attn.g_norm.bias".format,
    "blocks\.(\d+)\.ffn\.x_k": "model.layers.{}.ffn.x_k".format,
    "blocks\.(\d+)\.ffn\.key\.weight": "model.layers.{}.ffn.key.weight".format,
    "blocks\.(\d+)\.ffn\.value\.weight": "model.layers.{}.ffn.value.weight".format,
}


RWKVAG_INDEX = {
    "att.x_r": 0,
    "att.x_w": 1,
    "att.x_k": 2,
    "att.x_v": 3,
    "att.x_a": 4,
    "att.x_g": 5
}


def convert_pth_to_safetensors(input_path, output_path, dtype=None):
    # 加载.pth文件
    state_dict = torch.load(input_path, map_location="cpu")
    if dtype == "bfloat16":
        save_dtype = torch.bfloat16
    elif dtype == "float16":
        save_dtype = torch.float16
    elif dtype == "float32":
        save_dtype = torch.float32
    else:
        raise TypeError("not support dtype")

    # 如果加载的是整个模型而非state_dict，提取state_dict
    if isinstance(state_dict, torch.nn.Module):
        state_dict = state_dict.state_dict()

    default_dict = {}
    rwkvag_dict = {}
    for old_key, param in tqdm(state_dict.items()):
        param = param.contiguous()
        if param.dtype != save_dtype:
            param = param.to(save_dtype)

        if old_key == "head.weight":
            default_dict["lm_head.weight"] = param
        elif old_key == "emb.weight":
            default_dict["model.embeddings.weight"] = param
        elif old_key == "ln_out.weight":
            default_dict["model.norm.weight"] = param
        elif old_key == "ln_out.bias":
            default_dict["model.norm.bias"] = param

        else:
            success = False
            for old_key_pattern, new_key in WEIGHT_MAPPING.items():
                pattern = re.search(old_key_pattern, old_key)
                if pattern:
                    success = True
                    layer_idx = int(pattern.group(1))
                    new_key = WEIGHT_MAPPING[old_key_pattern](layer_idx)
                    lora_pattern = re.search("[awgv]_lora\.lora\.(\d+).weight", new_key)

                    if len(param.shape) == 3:
                        key_ends = ".".join(old_key.split(".")[-2:])

                        if key_ends in RWKVAG_INDEX:
                            index = RWKVAG_INDEX[key_ends]
                            if new_key not in rwkvag_dict:
                                rwkvag_dict[new_key] = torch.zeros(6, param.shape[-1])
                            rwkvag_dict[new_key][index, :] = param[0, 0, :]
                            continue
                        else:
                            param = param[0, 0, :]

                    elif lora_pattern:
                        param = param.t().contiguous()

                    default_dict[new_key] = param
                    break
            if not  success:
                raise ValueError(f"{old_key} pattern not found")

    default_dict.update(**rwkvag_dict)
    # 保存为safetensors格式
    save_file(default_dict, output_path)
    # model = AutoModelForCausalLM.from_pretrained("./", trust_remote_code=True)
    print(f"转换成功！文件已保存至：{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将.pth文件转换为safetensors格式")
    parser.add_argument("--input_path", type=str, default="/home/maijianqiang/RWKV-7-word/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth",
                        help="输入的.pth文件路径")
    parser.add_argument("--output_path", type=str, default="/home/maijianqiang/rwkv7-1.5B-world/model.safetensors",
                        help="输出的.safetensors文件路径")
    parser.add_argument("--dtype", type=str, default="float32", help="safetensors保存的数据类型")
    args = parser.parse_args()

    convert_pth_to_safetensors(args.input_path, args.output_path, args.dtype)
