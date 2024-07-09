from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

device = "cuda:1" # the device to load the model onto

path = "model_hub/qwen2_0.5B_instruct"

model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(path)

# peft_model_id = "output/"
# lora_config = PeftConfig.from_pretrained(peft_model_id)
# model = PeftModel.from_pretrained(model, peft_model_id, adapter_name="qwen2", config=lora_config)

model.to(device).eval()

# prompt = "输入三支篮球队的名称并生成一个适当的口号。输入：俄克拉荷马城雷霆队，芝加哥公牛队，布鲁克林网队。"
# prompt = "分类以下数字系列。输入：\n2、4、6、8"
prompt = "用“黎明”、“天空”和“广阔”这三个词组成一个句子。"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(text)

model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    do_sample=False,
    top_p=1.0,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)