from flask import Flask, request, jsonify
from dae_eval import DAEEvaluator

app = Flask(__name__)

# 加载模型
eval = DAEEvaluator()

# 定义路由
@app.route('/dae_doc', methods=['POST'])
def predict():
    # 从请求参数中获取输入数据
    data = request.json
    claim = data['claim']
    document = data['document']
    
    # 对输出数据进行后处理
    result = eval.score([document], [claim], use_tqdm=False)

    # 返回响应结果
    return jsonify(result[0])

if __name__ == '__main__':
    app.run(port=10007)