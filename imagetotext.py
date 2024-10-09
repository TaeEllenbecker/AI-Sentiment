'''
If you use this project or build upon it, please cite the following paper:

Plain Text:

Dai, W., Lee, N., Wang, B., Yang, Z., Liu, Z., Barker, J., Rintamaki, T., Shoeybi, M., Catanzaro, B., & Ping, W. (2024). NVLM: Open Frontier-Class Multimodal LLMs. arXiv preprint.

BibTeX:

BibTeX: @article{nvlm2024, title={NVLM: Open Frontier-Class Multimodal LLMs}, author={Dai, Wenliang and Lee, Nayeon and Wang, Boxin and Yang, Zhuolin and Liu, Zihan and Barker, Jon and Rintamaki, Tuomas and Shoeybi, Mohammad and Catanzaro, Bryan and Ping, Wei}, journal={arXiv preprint}, year={2024}}
'''

import torch
from transformers import AutoModel

path = "nvidia/NVLM-D-72B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    trust_remote_code=True).eval()
