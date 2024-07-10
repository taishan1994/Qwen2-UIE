# Qwen2-UIE
基于Qwen2模型进行通用信息抽取【实体/关系/事件抽取】。

该项目使用LLaMA-Factory作为训练的框架，Qwen2-0.5B-Instruct作为基础模型，yayi-uie作为训练数据，在1台8卡4090上训练了226000步。

# 模型训练

- 如果不需要训练，可以下载训练好的权重：https://www.modelscope.cn/models/xiximayou/Qwen2-UIE
- qwen模型权重下载：https://www.modelscope.cn/models/qwen/Qwen2-0.5B-Instruct
- yayi-uie数据下载：https://github.com/wenge-research/YAYI-UIE

```shell
1.进入到LLaMA-Factory
2.pip install -e ".[torch,metrics]"（碰到环境问题可以先看看LLaMA-Factory的仓库）
3.modelscope下载模型到model_hub/qwen2_0.5B_instruct下
4.yayi-uie下载数据，然后转换成data下面的yayi-sft.json里面的格式
5.执行指令
MODEL_PATH="model_hub/qwen2_0.5B_instruct"
NPROC_PER_NODE=8 # 使用的显卡数目，根据具体情况而定

NNODES=1
NODE_RANK=0
MASTER_ADDR="0.0.0.0"
MASTER_PORT=8896

DS_CONFIG_PATH="examples/deepspeed/ds_z2_config.json"
OUTPUT_PATH="output/qwen2_0.5B_uie"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
  "

NCCL_IB_DISABLE=1, NCCL_P2P_DISABLE=1 torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --flash_attn "auto" \
    --model_name_or_path $MODEL_PATH \
    --dataset alpaca_yayi_sft \
    --template qwen \
    --finetuning_type "full" \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 4096 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 10 \
    --bf16

```

# 模型预测

```python
import json
from threading import Thread

from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer

device = "cuda:0"  # the device to load the model onto

# path = "model_hub/qwen2_0.5B_instruct"
path = "output/qwen2_0.5B_uie/checkpoint-226000"

model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype="auto",
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(path)

prompt = "文本: AC米兰退出下赛季欧联杯，参加欧战的资格也就顺延给了本赛季排名在他们身后的罗马和都灵，对于红狼来说，他们可以跳过欧联杯资格赛，直接进入欧联杯小组赛，而都灵则得到了一个参加资格赛的名额。 \n【事件抽取】给定的事件类型列表是['财经/交易-出售/收购', '财经/交易-跌停', '财经/交易-加息', '财经/交易-降价', '财经/交易-降息', '财经/交易-融资', '财经/交易-上市', '财经/交易-涨价', '财经/交易-涨停', '产品行为-发布', '产品行为-获奖', '产品行为-上映', '产品行为-下架', '产品行为-召回', '交往-道歉', '交往-点赞', '交往-感谢', '交往-会见', '交往-探班', '竞赛行为-夺冠', '竞赛行为-晋级', '竞赛行为-禁赛', '竞赛行为-胜负', '竞赛行为-退赛', '竞赛行为-退役', '人生-产子/女', '人生-出轨', '人生-订婚', '人生-分手', '人生-怀孕', '人生-婚礼', '人生-结婚', '人生-离婚', '人生-庆生', '人生-求婚', '人生-失联', '人生-死亡', '司法行为-罚款', '司法行为-拘捕', '司法行为-举报', '司法行为-开庭', '司法行为-立案', '司法行为-起诉', '司法行为-入狱', '司法行为-约谈', '灾害/意外-爆炸', '灾害/意外-车祸', '灾害/意外-地震', '灾害/意外-洪灾', '灾害/意外-起火', '灾害/意外-坍/垮塌', '灾害/意外-袭击', '灾害/意外-坠机', '组织关系-裁员', '组织关系-辞/离职', '组织关系-加盟', '组织关系-解雇', '组织关系-解散', '组织关系-解约', '组织关系-停职', '组织关系-退出', '组织行为-罢工', '组织行为-闭幕', '组织行为-开幕', '组织行为-游行']，论元角色列表是['刑期', '领投方', '立案机构', '怀孕者', '失联者', '融资金额', '出生者', '产子者', '地点', '罢工人员', '原所属组织', '融资轮次', '会见主体', '致谢人', '庆祝方', '加息幅度', '加盟者', '开庭案件', '受伤人数', '加息机构', '约谈对象', '奖项', '发布方', '胜者', '裁员人数', '禁赛机构', '上映方', '坍塌主体', '死者', '死亡人数', '退役者', '被下架方', '订婚主体', '震级', '开庭法院', '赛事名称', '降价幅度', '举报对象', '交易物', '会见对象', '生日方年龄', '拘捕者', '解约方', '降价物', '被拘捕者', '被感谢人', '被解约方', '裁员方', '道歉对象', '被告', '求婚者', '降价方', '罢工人数', '夺冠赛事', '执法机构', '探班主体', '罚款对象', '探班对象', '死者年龄', '袭击对象', '收购方', '被解雇人员', '获奖人', '解散方', '跌停股票', '解雇方', '退赛赛事', '震源深度', '入狱者', '约谈发起方', '召回内容', '颁奖机构', '退赛方', '生日方', '退出方', '出售价格', '禁赛时长', '所加盟组织', '立案对象', '游行人数', '融资方', '活动名称', '出售方', '降息幅度', '上映影视', '袭击者', '游行组织', '涨停股票', '降息机构', '时间', '出轨方', '出轨对象', '离职者', '道歉者', '所属组织', '上市企业', '震中', '发布产品', '原告', '结婚双方', '涨价方', '点赞对象', '参礼人员', '下架产品', '涨价幅度', '被禁赛人员', '离婚双方', '涨价物', '跟投方', '点赞方', '晋级方', '晋级赛事', '求婚对象', '败者', '下架方', '召回方', '停职人员', '分手双方', '罚款金额', '举报发起方', '冠军']。根据事件类型列表和论元角色列表从给定的输入中抽取可能的事件。请以json[{'trigger':'', 'type':'', 'arguments': {角色:论元}},]的格式回答。\n答案："
messages = [
    # {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=2048,
    do_sample=False,
    top_p=1.0,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

```

其余类型的使用参考训练数据里面的格式。

# 补充

- 如何使用自己的数据集训练？在data下面构造yayi-sft.json格式的数据，然后在data下的dataset_info.json进行注册，最后修改训练指令中的数据集名称即可（dataset_info里面自定义的）。
- 目前已知的问题：模型对实体识别自定义标签不敏感，会自己输出其他的实体，如果是自己的数据集，可能需要再进一步进行微调训练。
- 请遵循相应项目的开源协议。
