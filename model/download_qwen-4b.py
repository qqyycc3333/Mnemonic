from transformers import AutoModelForCausalLM, AutoTokenizer
from openlm_hub import repo_download

# yuan = "meta-llama/Llama-3.2-3B"
yuan = "Qwen/Qwen3-4B-Base"


# model = AutoModelForCausalLM.from_pretrained(yuan)
# tokenizer = AutoTokenizer.from_pretrained(yuan)

save_path = "./model/qwen-4b"

# print(save_path)
# model.save_pretrained(save_path)
# tokenizer.save_pretrained(save_path)


####

model = AutoModelForCausalLM.from_pretrained(
    repo_download(yuan),
    # torch_dtype="auto",
    # device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(repo_download(yuan),trust_remote_code=True)
model = model.to("cpu")

print("save_path: ", save_path)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# # 单个文件下载
# from openlm_hub import download
# download(repo_id="Qwen/Qwen1.5-0.5B-Chat",filename='config.json')
# # 整个 repo 下载
# from openlm_hub import repo_download
# repo_download(repo_id="Qwen/Qwen1.5-0.5B-Chat")
# 文件默认将被下载到本地缓存文件夹。如~/.cache/openlm/hub/5e09bc6f20bc45b0a4e60be2eb37c449/
# 文件被默认下载到本地的缓存文件夹，用户也可以指定下载地址 (相同的下载地址不会重复下载且线程安全)
# from openlm_hub import repo_download
# repo_download(repo_id="Qwen/Qwen1.5-0.5B-Chat",local_dir='./')