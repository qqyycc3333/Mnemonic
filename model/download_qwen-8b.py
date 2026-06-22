from transformers import AutoModelForCausalLM, AutoTokenizer
from openlm_hub import repo_download

# yuan = "meta-llama/Llama-3.2-3B"
yuan = "Qwen/Qwen3-8B-Base"


# model = AutoModelForCausalLM.from_pretrained(yuan)
# tokenizer = AutoTokenizer.from_pretrained(yuan)

save_path = "./model/qwen-8b"

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
