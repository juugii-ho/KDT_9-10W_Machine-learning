import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression


def fcolumns (old, feature, data, new):
    new = old[old[feature] == data]
    new = new.reset_index(drop=True)
    return new

def finfo(DF):
    print(f'{DF.info()},\n')
    print(f'{DF.head(3)},\n')
    print(f'{DF.tail(3)},\n')
    print(DF.describe())

def fzdata(DF):
    mean_value = DF.mean()
    std_value = DF.std()
    print('mean_value, std_value : ', mean_value, std_value)
    z_data = (DF - mean_value) / std_value
    print()
    print()
    print('''# 양끝단 제거를 위한 기준값 임의로 잡기=> 1.0
# base = 1.0
# mask = z_data.abs()>base
# DF[~mask].dropna(inplace=True)''')
    print()
    print()

    return z_data

def fboxplot(DF):
    bp_obj = plt.boxplot(DF)
    print(f"""'최저 :', {bp_obj['whiskers'][0].get_ydata()[1]}, 
    '1QR :' {bp_obj['whiskers'][0].get_ydata()[0]}, 
    '3QR :' {bp_obj['whiskers'][1].get_ydata()[0]}
    '최고 :' {bp_obj['whiskers'][1].get_ydata()[1]})""")
    iqr = bp_obj['whiskers'][0].get_ydata()[0] - bp_obj['whiskers'][1].get_ydata()[0]
    print(f'IQR : {iqr}')




from sklearn.linear_model import LinearRegression

def fLinearTrainScore(X_train, X_test, y_train, y_test):
    '''
    model : 모델 객체 선형
    ScalerList : 스케일러 종류
    scaler : 스케일링 인수 획득 - 평균, 표준편차 등 획득
    STD : 평균 0, 편차 1, MM : 최소 0, 최대 1, RB : 중앙값0, IQR 1,
    train, test : 스케일링 진행 - 스케일링 인수 활용
    model.fit : 모델에 스케일링 값 학습
    train_score, test_score : 모델의 점수 평가
    return scaler
    '''
    model = LinearRegression()
    ScalerList = [StandardScaler(), MinMaxScaler(), RobustScaler()]
    for idx, i in enumerate(ScalerList):
        scaler = i
        scaler.fit(X_train)
        train = scaler.transform(X_train)
        test = scaler.transform(X_test)
        model.fit(train, y_train)
        train_score = model.score(train, y_train)
        test_score = model.score(test, y_test)
        pred_test = model.predict(test)
        print(f'{idx+1}) {i} : Train_score : {train_score*100:.3f}% --- Test Score : {test_score*100:.3f}%', end=' ')
        print(f'\t Fit :  {(test_score-train_score)*100:.3f}%', end='')
        if train_score > test_score:
            print('(과대적합)')
        elif test_score<50 and train_score < 50:
            print('(과소적합)')
        else:
            print()
        print(f'\tPredict_score : {test_score*100:.3f}%')


def fPlot(DF):
    plt.plot(DF, label=DF.columns)
    plt.legend()
    plt.show()


def fPoly_Plot(DF):
    poly = PolynomialFeatures()
    plt.plot(DF, label=poly.get_feature_names_out())
    plt.legend()
    plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score