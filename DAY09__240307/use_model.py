## ---------------------------------------------------------------
## 모델 활용한 서비스 제공
## ---------------------------------------------------------------
# 모듈 로딩
from joblib import *

# 전역 변수
model_file = '../model/iris_dt.pkl'

# 모델 로딩
model = load(model_file)

# 로딩된 모델 확인
# print(model.classes_)

# 붓꽃 정보 입력 => 4개 피쳐
datas = input("붓꽃의 꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비 입력 : ")
if len(datas):
    datas_list = list(map(int, datas.split(',')))
    print(datas_list)

    # 입력된 정보에 해당하는 품종 알려주기
    # 모델의 predict(2D)
    pre_iris = model.predict([datas_list])
    proba = model.predict_proba([datas_list])
    print(f'해당 꽃은 {proba[0][-1]}% 확률로 {pre_iris}입니다.')

    print(f'해당 꽃은 {pre_iris}입니다.')

    # model.predict()
else:
    print('입력된 정보가 없습니다.')