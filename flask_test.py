import requests
import json

url_dic = {
    'cloze': 10000,
    'dae': 10002,
    'factcc': 10003,
    'feqa': 10004,
    'summacconv': 10006,
}
url_pre = 'http://localhost:'
for key in url_dic:
    data = {'document': 'Bob went to Beijing.', 'claim': 'Bob went to Beijing.'}
    # 发送POST请求并获取响应
    response = requests.post(url_pre+str(url_dic[key])+'/'+key, json=data)
    # 解析响应JSON数据
    result = json.loads(response.text)
    print(key)
    print(result)