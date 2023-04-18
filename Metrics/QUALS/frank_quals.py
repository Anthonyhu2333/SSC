from flask import Flask, request, jsonify
from quals_eval import QUALSEval

app = Flask(__name__)

# 加载模型
eval = QUALSEval()

# 定义路由
@app.route('/quals', methods=['POST'])
def predict():
    # 从请求参数中获取输入数据
    data = request.json
    
    # 对输出数据进行后处理
    result = eval.score(**data)

    # 返回响应结果
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=10005)