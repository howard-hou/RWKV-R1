
## Installation

To use this code, you need to have the following dependencies installed:

- torch 2.5.1
- cuda 12.1+
- python==3.11.10
- transformers==4.48.2
- latex2sympy2_extended==1.0.6

# pt to safetensors
python pt_to_safetensors.py --input_path {pth_dir}/xxx.pth --output_path {hf_dir}/model.safetensors

## 注意
以上软件版本要严格匹配，python< 3.11.10 triton会有报错
## Usage

1. Clone this repository.
git clone https://github.com/MaigeWhite/lighteval.git
2. pip install -e lighteval/

3. git clone https://github.com/fla-org/flash-linear-attention.git  # commit id dd5de5
4. cd flash-linear-attention && python setup.py bdist_wheel && pip install dist/*.whl

```bash
# default eval math_500
bash eval_rwkv.sh

```
