# 백엔드 라이브러리
from flask import Flask, request
from flask_cors import CORS
app = Flask(__name__)
CORS(app, supports_credentials=True)
# 인공지능 라이브러리
import torch
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("Huffon/sentence-klue-roberta-base")


@app.route("/")
def main():
  return {'name' : 'here is main'}
docs = []

global res_datas
res_datas = [  {
  "id":"admin",
  "idx":0,
  "score":0
},]
# 예측
@app.route("/predict", methods=['GET', 'POST'])
def predict():
  global res_score
  if request.method == 'POST':
    answer = request.get_json()
    print(answer['answer'])
    docs.append(answer['answer'])

    document_embeddings = model.encode(docs)
    query = ["네 지원할게요!", "아니요 잘 몰라요...", "아니요... 공부 도와주세요!", 
    "잠깐 밖에서 쉴래요?", "저도 이제 가려고요! 같이 가요!", "빗자루질 할래요!", "유튜브 봐요!"
    ,"꽃밭 가요!", "같이 사진 찍어요!", "너무 잘 나왔어요"]

    # 데이터 추가 작업
    diff = 0
    total_len = len(res_datas)
    for data in res_datas:
      if data['id'] == answer['id']:
        print("이 아이디는 존재합니다.")
        break
      else:
        diff+=1
    if diff == total_len:
      print("추가합니다")
      up_dict = {}
      up_dict['id'] = answer['id']
      up_dict['idx'] = 0
      up_dict['score'] = 0
      res_datas.append(up_dict)

    cur_situation = res_datas[diff]['idx']
    print(f"{cur_situation}번 상황\n")
    query_embedding = model.encode(query[cur_situation])

    top_k = min(5, len(docs))

    # 입력 문장 - 문장 후보군 간 코사인 유사도 계산 후,
    cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]

    # 코사인 유사도 순으로 `top_k` 개 문장 추출
    top_results = torch.topk(cos_scores, k=top_k)

    print(f"입력 문장: {query[cur_situation]}")
    
    for i, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
      print(f"{i+1}: {docs[idx]} {'(유사도: {:.4f})'.format(score)}\n")

      res_datas[diff]['idx'] += 1

      if(score >= 0.5):
        res_datas[diff]['score'] += 1
        return "True"
      else:
        res_datas[diff]['score'] -= 1
        return "False"

@app.route("/result", methods=['POST'])
def result():
  answer = request.get_json()
  str_res = "nothing"
  for data in res_datas:
    if data['id'] == answer['id']:
      str_res = str(data['score'])
  return str_res

@app.route("/all_result", methods=['GET'])
def all_result():
  return res_datas

if __name__ == "__main__":
  app.run(host = '0.0.0.0', port=6006, debug=True)
