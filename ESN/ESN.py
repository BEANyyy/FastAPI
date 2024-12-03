import numpy as np
from ESN.pyESN import ESN
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed


# ESN모델 생성 후 예측 값 반환 함수
def create_esn_model(sparsity, spectral_radius, noise, train_X, train_Y, test_X):

    # ESN 기본 값 설정
    n_reservoir = 1000
    sparsity = sparsity
    rand_seed = 23
    spectral_radius = spectral_radius
    noise = noise

    # 각 입력벡터(row)가 8개의 열을 가짐 => 입력데이터의 차원은 8
    n_samples, n_features = train_X.shape

    # 모델 생성
    esn = ESN(n_inputs=n_features,  # 입력 차원의 수(=입력 데이터의 열 수)
              n_outputs=1,  # 출력 차원의 수
              n_reservoir=n_reservoir,  # reservoir 뉴런의 수
              sparsity=sparsity,  # recurrent weights 중 0으로 설정되는 비율
              random_state=rand_seed,  # 난수 생성을 위한 시드값
              spectral_radius=spectral_radius,  # recurrent weight matrix의 스펙트럴 반지름(고유값 중 최대 값)
              noise=noise)  # 각 뉴런에 추가되는 노이즈(정규화) -> 과적합방지

    # 훈련 데이터 설정
    trainlen = len(train_X)  # 훈련 데이터의 길이
    futureTotal = len(test_X)  # 예측할 미래 데이터의 전체 수
    pred_tot = np.zeros(futureTotal)  # 예측결과를 저장할 배열

    # 훈련 데이터로 ESN 모델 훈련
    esn.fit(train_X, train_Y)

    # 예측 수행
    pred_tot = esn.predict(test_X)
    pred_tot = pred_tot.flatten()

    # backtesting1, 2에 사용에 필요한 것 반환
    return pred_tot


# 트렌드 데이터 학습-검증 분할 함수
def ts_train_test(df):
    all_data = df
    # 독립변수 선택
    features = ['RSI_sig', 'SMA_sig', 'ROC_sig', 'DPO_sig', 'STOCH_sig', 'MACD_sig', 'GDC_sig']
    data = all_data[features].values
    # 종속변수 선택
    target = all_data['TREND'].values
    # train/test으로 3:7 이용
    N = len(all_data)
    traintest_cutoff = int(np.ceil(0.7 * N))
    # 학습 데이터 검증데이터 추출
    train_X, train_Y = data[:traintest_cutoff], target[:traintest_cutoff]
    test_X, test_Y = data[traintest_cutoff:], target[traintest_cutoff:]

    # Convert data types to float64
    train_X = train_X.astype(np.float64)
    train_Y = train_Y.astype(np.float64)

    return train_X, train_Y, test_X, test_Y


# # ESN모델 파라미터 최적화 함수(백테스팅 기준: 수익률)
# def best_fit_esn_model(train_X, train_Y, test_X, test_Y, all_data_orign_30):
#     # 조정할 파라미터들
#     sparsity_set = [0.1, 0.2, 0.3, 0.4]
#     radius_set = [0.9, 1.3, 1.7]  # 순환 가중치 행렬의 스펙트럴 반지름 설정
#     noise_set = [0.001, 0.01, 0.1]  # 각 뉴런에 추가되는 노이즈 설정
#
#     # 설정된 값들의 크기 계산
#     sparsity_set_size = len(sparsity_set)
#     radius_set_size = len(radius_set)
#     noise_set_size = len(noise_set)
#
#     # MSE 값 저장할 3D 배열 초기화
#     loss = np.zeros([sparsity_set_size, radius_set_size, noise_set_size])
#
#     # tqdm을 활용한 진행률 표시
#     total_iterations = sparsity_set_size * radius_set_size * noise_set_size
#     progress_bar = tqdm(total=total_iterations, desc="ESN 파라미터 최적화 중")
#
#
#     # sparsity, 반지름, 노이즈 설정 반복
#     for s in range(sparsity_set_size):
#         sparsity = sparsity_set[s]
#         for l in range(radius_set_size):
#             rho = radius_set[l]
#             for j in range(noise_set_size):
#                 noise = noise_set[j]
#
#                 # 해당 파라미터조건으로 모델 생성 및 예측 결과 출력
#                 pred_tot = create_esn_model(sparsity, rho, noise, train_X, train_Y, test_X)
#
#                 # 시그널 생성
#                 all_data_orign_30 = making_sig(pred_tot, test_Y, all_data_orign_30)
#
#                 # 현재 설정에 모델 평가(백테스팅) 및 결과 저장
#                 loss[s, l, j] = result_profit(all_data_orign_30, 'pred_sig')
#
#                 # tqdm 진행률 업데이트
#                 progress_bar.update(1)
#
#     progress_bar.close()
#
#     max_profit = np.max(loss)
#     max_indices = np.where(loss == max_profit)
#     best_sparsity = sparsity_set[max_indices[0][0]]
#     best_rho = radius_set[max_indices[1][0]]
#     best_noise = noise_set[max_indices[2][0]]
#
#
#     return max_profit, best_sparsity, best_rho, best_noise

def best_fit_esn_model(train_X, train_Y, test_X, test_Y, all_data_orign_30):
    sparsity_set = [0.1, 0.2, 0.3, 0.4]
    radius_set = [0.9, 1.3, 1.7]
    noise_set = [0.001, 0.01, 0.1]

    sparsity_set_size = len(sparsity_set)
    radius_set_size = len(radius_set)
    noise_set_size = len(noise_set)

    total_iterations = sparsity_set_size * radius_set_size * noise_set_size
    loss = np.zeros([sparsity_set_size, radius_set_size, noise_set_size])

    # 병렬 처리를 위한 작업 함수
    def evaluate_params(s, l, j):
        sparsity = sparsity_set[s]
        rho = radius_set[l]
        noise = noise_set[j]
        pred_tot = create_esn_model(sparsity, rho, noise, train_X, train_Y, test_X)
        all_data_orign_30_local = making_sig(pred_tot, test_Y, all_data_orign_30.copy())
        profit = result_profit(all_data_orign_30_local, 'pred_sig')
        return (s, l, j, profit)

    # tqdm_joblib으로 tqdm과 병렬 처리를 함께 사용
    with tqdm_joblib(tqdm(total=total_iterations, desc="ESN 파라미터 최적화 중", leave=True)):
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_params)(s, l, j)
            for s in range(sparsity_set_size)
            for l in range(radius_set_size)
            for j in range(noise_set_size)
        )

    # 결과 저장
    for s, l, j, profit in results:
        loss[s, l, j] = profit

    max_profit = np.max(loss)
    max_indices = np.where(loss == max_profit)
    best_sparsity = sparsity_set[max_indices[0][0]]
    best_rho = radius_set[max_indices[1][0]]
    best_noise = noise_set[max_indices[2][0]]

    return max_profit, best_sparsity, best_rho, best_noise



# backtesting1에 사용될 시그널 생성, backtesting2에 사용될 시그널 생성하여 all_data_orign_30에 붙이기
def making_sig(pred_tot, test_Y, all_data_orign_30):
    # 일단 예측 데이터 붙여
    all_data_orign_30['pred'] = pred_tot
    # 예측값을 시그널로 변경
    # 조건에 따라 pred_sig 열을 추가
    all_data_orign_30['pred_sig'] = all_data_orign_30['pred'].apply(
        lambda x: 0 if -0.7 < x < 0.7 else (1 if x >= 0.7 else -1))
    all_data_orign_30['pred_sig'] = all_data_orign_30['pred_sig'].astype(
        int)  # 시그널만 int형으로 고치기(수익률 계산 함수에서 오류없이 사용하기 위함)

    # 인덱스 재편성
    all_data_orign_30.reset_index(drop=True, inplace=True)

    # 일단 실제 정답값 붙여
    all_data_orign_30['trend_30'] = test_Y

    # 조건에 따라 pred_sig 열을 추가
    all_data_orign_30['trend_30_sig'] = all_data_orign_30['trend_30'].apply(
        lambda x: 0 if -0.7 < x < 0.7 else (1 if x >= 0.7 else -1))
    all_data_orign_30['trend_30_sig'] = all_data_orign_30['trend_30_sig'].astype(
        int)  # 시그널만 int형으로 고치기(수익률 계산 함수에서 오류없이 사용하기 위함)

    return all_data_orign_30


# 시그널 수익률 계산 함수
def result_profit(df, sig_column):
    
    # 초기 자본금 설정
    initial_capital = 1000

    # 보유 주식 수와 자본금 추적
    shares_held = 0
    capital = initial_capital
    capital_history = [capital]

    # 매수, 매도, 또는 보유 결정에 따른 자본금 변화 계산
    for i in range(1, len(df)):
        if df[sig_column][i] == 1:  # Buy 시그널인 경우
            shares_to_buy = capital // df['Close'][i]  # 보유 가능한 주식 수 계산
            shares_held += shares_to_buy
            capital -= shares_to_buy * df['Close'][i]

        elif df[sig_column][i] == -1:  # Sell 시그널인 경우
            capital += shares_held * df['Close'][i]  # 보유 주식 매도
            shares_held = 0

        if df[sig_column][i] == 0:  # 0 시그널인 경우 (보유 유지)
            capital_history.append(capital + shares_held * df['Close'][i])

        capital_history.append(capital + shares_held * df['Close'][i])  # 자본금 변화 추적

    # 수익률 계산
    returns = (capital_history[-1] - initial_capital) / initial_capital * 100
    # print("예측한 결과에 대한 수익률 : ", returns)
    return returns


# 변경된 모듈화에 맞게 일치율 함수 변경 -> 최적화 함수에 수익률 구하는데 옆에 넣어서 더해
def match_backtest(all_data_orign_30):
    # 두 열을 비교하여 일치하는 횟수를 계산(일치하면 1 아니면 0)
    all_data_orign_30['match_count'] = (all_data_orign_30['pred_sig'] == all_data_orign_30['trend_30_sig']).astype(int)

    # 일치하는 횟수를 백분율로 계산
    matching_percentage = (all_data_orign_30['match_count'].sum() / len(all_data_orign_30)) * 100
    return matching_percentage