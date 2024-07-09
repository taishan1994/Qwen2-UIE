import json

with open("train_sft.jsonl", "r") as fp:
    data = fp.readlines()
res = []
for d in data:
    d = json.loads(d)
    res.append(d)

print(len(res))
with open("yayi-sft.json", "w") as fp:
    fp.write(json.dumps(res, ensure_ascii=False, indent=2))