import requests
import json

# 发送HTTP请求

data = {'texts':["人增福寿年增岁", "风吹云乱天垂泪"],'use_gpu':False, 'beam_width':5}
headers = {"Content-type": "application/json"}
url = "http://192.168.124.73:8866/predict/ernie_gen_couplet"
r = requests.post(url=url, headers=headers, data=json.dumps(data))
print(json.dumps(data))
print(data)

# 保存结果
results = r.json()["results"]
for result in results:
    print(result)

