from sklearn.metrics import mean_absolute_error,r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np

def fplot_scatter(nrows, ncols, targetSR, featureDF):

    n = 1
    for col in featureDF.columns:
        plt.subplot(nrows, ncols, n)
        plt.scatter(targetSR, featureDF[col], label=f'corr = {targetSR.corr(featureDF[col]):.4f}')
        plt.xlabel(targetSR.name)
        plt.title(f'{col}')
        plt.ylabel(col)
        plt.xticks([])
        plt.legend()
        n += 1
    plt.tight_layout()
    plt.show()

def fplot_hist(nrows, ncols, featureDF):

    n = 1
    for col in featureDF.columns:
        plt.subplot(nrows, ncols, n)
        plt.hist(featureDF[col],edgecolor='k', color = 'yellow')
        plt.title(f'{col}')
        plt.ylabel(col)
        n += 1
    plt.tight_layout()
    plt.show()

def fplot_box(nrows, ncols,targetSR, featureDF):
    n = 1
    for i in featureDF.columns:
        plt.subplot(nrows, ncols, n)
        plt.boxplot(featureDF[i])
        plt.xlabel(targetSR.name)
        plt.title(f'{i}')
        n += 1
    plt.tight_layout()
    plt.show()

def fpre_find_outlier_z(df, hold = 1):
    for i in df.columns:
        mean = df[i].mean()
        std = df[i].std()
        z_score = (df[i] - mean) / std
        mask = z_score.abs() > hold
        print(f'z - {i}의 이상치 개수 : {z_score[mask].count()}')

def fpre_fill_outliers_z(sr, hold, fill_value):

    valid_value = ['mean', 'median']
    if fill_value not in valid_value:
        raise ValueError(f"score_standard must be one of {valid_value}")

    mean = sr.mean()
    std = sr.std()
    z_score = (sr - mean) / std
    mask = z_score.abs() <= hold

    sr_copy = sr.copy()

    if fill_value == 'mean':
        sr_copy[mask] = sr_copy.mean()
    elif fill_value == 'median':
        sr_copy[mask] = sr_copy.median()

    return sr_copy

def fpre_find_outlier_iqr(df,threshold = 1.5):
    for i in df.columns:
        q1 = df[i].quantile(0.25)
        q3 = df[i].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - iqr * threshold
        upper = q3 + iqr * threshold

        print(f'iqr - {i}의 이상치 개수 : {df[(df[i] < lower)&(df[i] > upper)].count()}')

def fpre_fill_outliers_iqr(sr, threshold, fill_value):

    valid_value = ['mean', 'median']
    if fill_value not in valid_value:
        raise ValueError(f"score_standard must be one of {valid_value}")

    q1 = sr.quantile(0.25)
    q3 = sr.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr
    upper = q3 + iqr

    sr_copy = sr.copy()

    if fill_value == 'mean':
        sr_copy[(sr_copy < lower) & (sr_copy > upper)] = sr_copy.mean()
    elif fill_value == 'median':
        sr_copy[(sr_copy < lower) & (sr_copy > upper)] = sr_copy.median()

    return sr_copy


def fml_find_random_state(featureDF,targetSR):
    # 최적 random_state 값
    random_state_list = []
    for i in range(1,51):
        xtrain,xtest,ytrain,ytest = train_test_split(featureDF,targetSR,test_size=0.2,random_state=i)
        scaler = StandardScaler() # scaler 종류에 따른 큰 차이 없음
        scaler.fit(xtrain)
        xtrain_scaled = scaler.transform(xtrain)
        xtest_scaled = scaler.transform(xtest)
        model = LinearRegression() # model 종류에 따라 차이남
        model.fit(xtrain_scaled,ytrain)
        model.score(xtest_scaled,ytest)
        random_state_list.append(model.score(xtest_scaled,ytest))
    max_score = max(random_state_list)
    print(f'radom_state = {random_state_list.index(max_score)+1}\nscore : {max_score}')

    max_random_state = random_state_list.index(max_score)+1
    return max_random_state

def fml_find_maxK_re(xtrain,ytrain,xtest,ytest):
    max_k = xtrain.shape[0]
    test_scoreList = []
    train_scoreList = []

    for k in range(1, max_k + 1):
        knn_model = KNeighborsRegressor(n_neighbors=k)
        knn_model.fit(xtrain, ytrain)
        train_scoreList.append(knn_model.score(xtrain, ytrain))
        test_scoreList.append(knn_model.score(xtest, ytest))
    max_idx = test_scoreList.index(max(test_scoreList)) + 1
    K = max_idx
    return K

def find_maxK_cl(xtrain,ytrain,xtest,ytest):
    max_k = xtrain.shape[0]
    test_scoreList = []
    train_scoreList = []

    for k in range(1, max_k + 1):
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(xtrain, ytrain)
        train_scoreList.append(knn_model.score(xtrain, ytrain))
        test_scoreList.append(knn_model.score(xtest, ytest))
    max_idx = test_scoreList.index(max(test_scoreList)) + 1
    K = max_idx
    return K

def find_re_model(xtrain, ytrain, xtest, ytest, score_standard ):

    valid_scores = ['r2', 'mae', 'mse', 'rmse']
    if score_standard not in valid_scores:
        raise ValueError(f"score_standard must be one of {valid_scores}")

    models = [KNeighborsRegressor(),LinearRegression(),Ridge(),Lasso()]
    scalers = [StandardScaler(), MinMaxScaler(), RobustScaler()]

    acDict = {}
    model_score = {'model': [], 'scaler': [], 'train_score': [], 'test_score': [], 'r2': [], 'mae': [], 'mse': [],
                   'rmse': []}

    for scaler in scalers:
        scaler.fit(xtrain)
        scaled_xtrain = scaler.transform(xtrain)
        scaled_xtest = scaler.transform(xtest)

        for model in models:
            if isinstance(model, KNeighborsRegressor):
                print('----------------탐색중------------------')
                max_k = xtrain.shape[0]
                test_scoreList = []
                train_scoreList = []

                for k in range(1, max_k + 1):
                    knn_model = KNeighborsRegressor(n_neighbors=k)
                    knn_model.fit(xtrain, ytrain)
                    train_scoreList.append(knn_model.score(xtrain, ytrain))
                    test_scoreList.append(knn_model.score(xtest, ytest))
                max_idx = test_scoreList.index(max(test_scoreList)) + 1
                K = max_idx
                model = KNeighborsRegressor(n_neighbors = K)
            elif isinstance(model, Ridge):
                alphas = np.arange(0.1, 30., 0.1).tolist()
                scorelist = [[], []]
                for a in alphas:
                    model = Ridge(alpha=a, max_iter=30000)

                    model.fit(xtrain, ytrain)

                    scorelist[0].append(model.score(xtrain, ytrain))
                    scorelist[1].append(model.score(xtest, ytest))
                best_alpha = alphas[scorelist[1].index(max(scorelist[1]))]
                model = Ridge(alpha=best_alpha, max_iter=30000)
            elif isinstance(model, Lasso):
                alphas = np.arange(0.1, 30., 0.1).tolist()
                scorelist = [[], []]
                for a in alphas:
                    model = Lasso(alpha=a, max_iter=30000)

                    model.fit(xtrain, ytrain)

                    scorelist[0].append(model.score(xtrain, ytrain))
                    scorelist[1].append(model.score(xtest, ytest))
                best_alpha = alphas[scorelist[1].index(max(scorelist[1]))]
                model = Lasso(alpha=best_alpha, max_iter=30000)

            # KNN,Ridge,Laaso 모델이 아닐 때, 바로 아래 코드 수행
            model.fit(scaled_xtrain, ytrain)
            print(f'model : {model}')

            train_score = model.score(scaled_xtrain, ytrain)
            test_score = model.score(scaled_xtest, ytest)
            print(f'scaler : {scaler}\nTrain score : {train_score}\nTest score : {test_score}')

            y_pre = model.predict(scaled_xtest)
            r2 = r2_score(ytest, y_pre)
            mse = mean_squared_error(ytest, y_pre)
            mae = mean_absolute_error(ytest, y_pre)
            rmse = mean_squared_error(ytest, y_pre, squared=False)
            print(f'''
    [모델 설명도]\nR2 : {r2}\n[에러]\nMAE : {mae}\nMSE : {mse}\nRMSE : {rmse}\n--------------------------------------
    ''')

            acDict[(model, scaler)] = [train_score,test_score, r2, mae, mse, rmse]

            model_score['model'].append(model)
            model_score['scaler'].append(scaler)
            model_score['train_score'].append(train_score)
            model_score['test_score'].append(test_score)
            model_score['r2'].append(r2)
            model_score['mae'].append(mae)
            model_score['mse'].append(mse)
            model_score['rmse'].append(rmse)

    if score_standard == 'r2':
        max_ac = max(acDict, key=lambda k: acDict[k][2])
    elif score_standard == 'mae':
        max_ac = min(acDict, key=lambda k: acDict[k][3])
    elif score_standard == 'mse':
        max_ac = min(acDict, key=lambda k: acDict[k][4])
    elif score_standard == 'rmse':
        max_ac = min(acDict, key=lambda k: acDict[k][5])
    else:
        pass

    print(
        f'[최적의 모델] :{max_ac}\nTrain score : {acDict[max_ac][0]}\nTest score : {acDict[max_ac][1]} \nR2 : {acDict[max_ac][2]}'
        f'\nMAE : {acDict[max_ac][3]}\nMSE : {acDict[max_ac][4]}\nRMSE : {acDict[max_ac][5]}')
    return model_score

def find_scaler(xtrain, ytrain, xtest, ytest, model):
    scalers = [StandardScaler(), MinMaxScaler(), RobustScaler()]

    train_scores = []
    test_scores = []
    for scaler in scalers:
        scaler.fit(xtrain)
        scaled_xtrain = scaler.transform(xtrain)
        scaled_xtest = scaler.transform(xtest)

        model.fit(scaled_xtrain, ytrain)
        print(f'model : {model}')

        train_score = model.score(scaled_xtrain, ytrain)
        test_score = model.score(scaled_xtest, ytest)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print(f'scaler : {scaler}\nTrain score : {train_score}\nTest score : {test_score}')

        y_pre = model.predict(scaled_xtest)
        r2 = r2_score(ytest, y_pre)
        mse = mean_squared_error(ytest, y_pre)
        mae = mean_absolute_error(ytest, y_pre)
        rmse = mean_squared_error(ytest, y_pre, squared=False)
        print(f'''
[모델 설명도]\nR2 : {r2}\n[에러]\nMAE : {mae}\nMSE : {mse}\nRMSE : {rmse}\n\n--------------------------------------
''')


def find_poly_p(xtrain, ytrain, xtest, ytest):
    # poly 최적의 파라미터 값 찾기
    max_score = []
    for b in [True, False]:
        for d in range(1, 6):
            poly = PolynomialFeatures(interaction_only=b, degree=d)
            poly.fit(xtrain)
            xtrain_transformed = poly.transform(xtrain)
            xtest_transformed = poly.transform(xtest)

            model = LinearRegression()
            model.fit(xtrain_transformed, ytrain)
            score = model.score(xtest_transformed, ytest)
            print(b, d, score)
            max_score.append([b, d, score])

    max_element = max(max_score, key=lambda x: x[2])

    b_max, d_max, score_max = max_element[0], max_element[1], max_element[2]

    print(f'max score =>\ninteraction_only = {b_max}, degree = {d_max}, score = {score_max}')
    return b_max, d_max

def find_alpha(xtrain, ytrain, xtest, ytest):
    alphas = np.arange(0.1,30.,0.1).tolist()
    scorelist = [[], []]
    for a in alphas:
        model = Ridge(alpha=a, max_iter=30000)

        model.fit(xtrain, ytrain)

        scorelist[0].append(model.score(xtrain, ytrain))
        scorelist[1].append(model.score(xtest, ytest))
    best_alpha = alphas[scorelist[1].index(max(scorelist[1]))]
    return best_alpha

def print_alpha_plot(alphas,best_alpha,scorelist):
    plt.plot(alphas, scorelist[0], label='Train')
    plt.plot(alphas, scorelist[1], label='Test')
    plt.axvline(best_alpha, color='red', linestyle=':', label=f'alpha ={best_alpha}')
    plt.text(best_alpha + 1, 0.984, f'Best_alpha ={best_alpha}')
    plt.legend()
    plt.title('[Train & Test]')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge

def missing_value_and_outlier_detection(df):
  """
  결측치와 이상치를 확인하는 함수

  Args:
    df: 데이터프레임

  Returns:
    결측치 개수, 이상치 개수
  """
  missing_value_count = df.isnull().sum()
  outlier_count = 0
  print(1)
  for col in df.columns:
    q1, q3 = df[col].quantile([0.25, 0.75])
    print(2)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    print(3)
    outlier_count += (df[col] < lower_bound).sum() + (df[col] > upper_bound).sum()
    print(missing_value_count, outlier_count)
  return missing_value_count, outlier_count

def data_visualization(df):
  """
  데이터 시각화 함수

  Args:
    df: 데이터프레임

  Returns:
    산포도, 히트맵, 박스플롯, 플롯
  """
  for col in df.columns:
    plt.scatter(df[col], df['target'])
    plt.title(f'산포도: {col} vs target')
    plt.show()

  plt.matshow(df.corr())
  plt.title('히트맵')
  plt.show()

  for col in df.columns:
    plt.boxplot(df[col])
    plt.title(f'박스플롯: {col}')
    plt.show()

  plt.plot(df['target'])
  plt.title('타겟 변수 플롯')
  plt.show()

def model_selection(df, X_train, y_train, X_test, y_test):
  """
  모델 선택 함수

  Args:
    df: 데이터프레임
    X_train: 훈련 데이터
    y_train: 훈련 라벨
    X_test: 테스트 데이터
    y_test: 테스트 라벨

  Returns:
    최고 모델, 최고 모델 파라미터, 최고 모델 하이퍼 파라미터
  """
  models = {
      'LinearRegression': LinearRegression(),
      'Lasso': Lasso(),
      'Ridge': Ridge(),
      'BayesianRidge': BayesianRidge()
  }
  best_model = None
  best_params = None
  best_score = 0
  for model_name, model in models.items():
    for param in model.get_params():
      for value in np.linspace(0.001, 1, 100):
        model.set_params(**{param: value})
        score = cross_val_score(model, X_train, y_train, cv=5).mean()
        if score > best_score:
          best_score = score
          best_model = model_name
          best_params = model.get_params()
  return best_model, best_params, best_score

def feature_importance(df, model):
  """
  피처 중요도 분석 함수

  Args:
    df: 데이터프레임
    model: 학습된 모델

  Returns:
    피처 중요도 순위
  """
  feature_importance = pd.Series(model.coef_, index=df.columns)
  return feature_importance.sort_values(ascending=False)




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

# 데이터 불러오기
df = pd.read_csv('data/iris.csv')

# df =

# 결측치 및 이상치 확인
missing_value_and_outlier_detection(df)
#
# # 전처리 전 데이터 시각화
# data_visualization(df)
#
# # 결측치 처리 및 스케일링
# df.fillna(df.mean(), inplace=True)
# scaler = StandardScaler()
# df = scaler.fit_transform(df)
#
# # 레이블 변수 처리
# df['iris_type'] = df['iris_type'].astype('category')
# y = df['iris_type'].cat.codes
#
# # 훈련 데이터와 테스트 데이터 분할
# X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
#
# # 모델 목록
# all_estimators = {
#     'LinearRegression': LinearRegression(),
#     'Lasso': Lasso(),
#     'Ridge': Ridge(),
#     'BayesianRidge': BayesianRidge(),
#     'LogisticRegression': LogisticRegression(),
#     'SVM': SVC(),
#     'KNN': KNeighborsClassifier(),
#     'DecisionTreeClassifier': DecisionTreeClassifier(),
#     'RandomForestClassifier': RandomForestClassifier()
# }
#
# # 파라미터 및 하이퍼 파라미터 딕셔너리
# param_grid = {
#     'LinearRegression': {
#         'fit_intercept': [True, False],
#         'normalize': [True, False]
#     },
#     'Lasso': {
#         'alpha': np.linspace(0.001, 1, 100)
#     },
#     'Ridge': {
#         'alpha': np.linspace(0.001, 1, 100)
#     },
#     'BayesianRidge': {
#         'alpha_1': np.linspace(0.001, 1, 100),
#         'lambda_1': np.linspace(0.001, 1, 100)
#     },
#     'LogisticRegression': {
#         'C': np.linspace(0.001, 1, 100),
#         'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag']
#     },
#     'SVM': {
#         'C': np.linspace(0.001, 1, 100),
#         'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
#     },
#     'KNN': {
#         'n_neighbors': range(1, 20),
#         'weights': ['uniform', 'distance']
#     },
#     'DecisionTreeClassifier': {
#         'max_depth': range(1, 20),
#         'min_samples_split': range(2, 20)
#     },
#     'RandomForestClassifier': {
#         'n_estimators': range(1, 200),
#         'max_depth': range(1, 20)
#     }
# }
#
# # 랜덤 서치를 통한 최적의 모델, 파라미터, 하이퍼 파라미터 찾기
# best_model = None
# best_params = None
# best_score = 0
# for model_name, model in all_estimators.items():
#     param_grid_model = param_grid[model_name]
#     random_search = RandomizedSearchCV(model, param_grid_model, cv=5, n_iter=100)
#     random_search.fit(X_train, y_train)
#     score = random_search.best_score_
#     if score > best_score:
#         best_score = score
#         best_model = model_name
#         best_params = random_search.best_params_
#
# # 결과 출력
# print(f'최고 모델: {best_model}')
# print(f'최고 파라미터: {best_params}')
