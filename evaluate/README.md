
## Installation

To use this code, you need to have the following dependencies installed:

- python 3.11
- torch 2.5.1
- cuda 12.1+

## Usage

1. Clone this repository.
git clone https://github.com/MaigeWhite/lighteval.git
2. pip install -e lighteval/

3. git clone https://github.com/fla-org/flash-linear-attention.git
4. cd flash-linear-attention && python setup.py bdist_wheel && pip install dist/*.whl

```bash
# default eval math_500
bash eval_rwkv.sh

```
