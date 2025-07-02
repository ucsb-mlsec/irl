import json


with open('/home/kaijie/irl/data/expert_demo_s1k_save.json', 'r') as f:
    data1 = json.load(f)

with open('/home/kaijie/irl/data/expert_demo_s1k.json', 'r') as f:
    data2 = json.load(f)

print(len(data1))
print(len(data2))

print(data1[0]["question"])
print(data2[0]["question"])

data = []

for d1 in data1:
    question = d1["question"]
    for d2 in data2:
        if d2["question"] == question:
            d1["responses"] += d2["responses"]
            assert len(d1["responses"]) == 10

            data.append(d1)
    
print(len(data))
with open('/home/kaijie/irl/data/expert_demo_s1k_merged.json', 'w') as f:
    json.dump(data, f, indent=4)
