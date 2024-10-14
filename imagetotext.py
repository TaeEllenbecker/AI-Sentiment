import os

from huggingface_hub import hf_hub_download

HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

model_id = "stepfun-ai/GOT-OCR2_0"

filenames = ["config.json","generation_config.json","got_vision_b.py","model.safetensors","modeling_GOT.py","qwen.tiktoken",
"render_tools.py","special_tokens_map.json","tokenization_qwen.py","tokenizer_config.json"
]

for filename in filenames:
    downloaded_model_path = hf_hub_download(repo_id = model_id, filename = filename, token = HUGGING_FACE_API_KEY)

print(downloaded_model_path)


from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()


# input your test image
image_file = 'product-manager-business-card.jpg'

# plain texts OCR
res = model.chat(tokenizer, image_file, ocr_type='ocr')

# format texts OCR:
# res = model.chat(tokenizer, image_file, ocr_type='format')

# fine-grained OCR:
# res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_box='')
# res = model.chat(tokenizer, image_file, ocr_type='format', ocr_box='')
# res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_color='')
# res = model.chat(tokenizer, image_file, ocr_type='format', ocr_color='')

# multi-crop OCR:
# res = model.chat_crop(tokenizer, image_file, ocr_type='ocr')
# res = model.chat_crop(tokenizer, image_file, ocr_type='format')

# render the formatted OCR results:
# res = model.chat(tokenizer, image_file, ocr_type='format', render=True, save_render_file = './demo.html')

print(res)