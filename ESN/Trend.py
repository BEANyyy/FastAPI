# 1. FNC_Func_PST

import numpy as np
from scipy.signal import argrelextrema

def FNC_Func_PST(T, P, c):
    # c : 데이터 자체
    c = np.array(c, dtype=float)
    n = len(c)  # 전체 데이터의 길이를 구함

    c_min = argrelextrema(c, np.less)[0]  # c 데이터 국소 최솟값의 인덱스
    c_max = argrelextrema(c, np.greater)[0]  # c 데이터 국소 최댓값의 인덱스

    cp1 = np.concatenate((c_max, c_min))  # cp1 : 최댓값 최솟값의 위치를 병합한 배열
    cp2 = np.concatenate((c[c_max], c[c_min]))  # cp2 : 최댓값 최솟값 자체를 병합한 배열
    cp = np.column_stack((cp1[np.argsort(cp1)], cp2[np.argsort(cp1)])) # 각각 정렬 후 [cp1, cp2]로 병합

    if len(cp) == 0:  # cp 배열이 비어있다면 빈 리스트와 0을 반환하고 종료
        return [], 0

    index = 0
    sp = np.zeros((len(cp), 2)) # 2차원 배열 생성
    sp[index] = cp[index] # sp에 첫 번째 국소 최댓값/최솟값의 위치와 값 저장

    i = 1
    while i < len(cp) - 1:
        # 최댓값/최솟값의 위치와 현재 위치 사이의 간격이 T보다 작거나
        # 현재 위치와 다음 위치 사이의 값 차이가 평균값에 대비하여 P보다 작으면 스킵.
        if (cp[i+1,0] - cp[i,0] < T) and (abs(cp[i+1,1] - cp[i,1]) / ((cp[i,1] + cp[i+1,1]) / 2) < P):
            i += 2

        # 위의 두 조건에 해당하지 않는다면 현재 위치를 sp에 저장하고 인덱스 +!
        else:
            index += 1
            sp[index] = cp[i]
            i += 1

    # 스킵되지 않은 값들만 포함되도록 sp 자르기
    sp = sp[:index+1]

    # 만약  sp 배열의 길이가 1이라면
    if len(sp) == 1:
        # 결과 배열을 0으로 채우고
        out_arg1 = np.zeros(n)
        # out_arg2를 0으로 설정 후 함수 종료
        out_arg2 = 0

    # sp 배열의 길이가 1이 아닌 경우
    else:
        # sp 배열에서 값 추출하여 temp에 저장
        temp = sp[:,1]
        # temp 배열에서 국소 최소값과 국소 최대값의 위치 찾기
        temp_min = argrelextrema(temp, np.less)[0]
        temp_max = argrelextrema(temp, np.greater)[0]

        # 최소값 최대값의 위치를 병합
        t1 = np.concatenate((temp_max, temp_min))
        t2 = np.concatenate((temp[temp_max], temp[temp_min]))

        # 위치 + 해당 값 배열 생성
        rp_t = np.column_stack((sp[t1[np.argsort(t1)],0], t2[np.argsort(t1)]))

        if len(rp_t) == 0:
            return [], 0

        # rp_t와 동일한 구조를 가지고 0으로 채워진 rp 배열 생성
        rp = np.zeros_like(rp_t)

        # 처음 두 개 값 설정
        rp[0,0] = rp_t[0,0]
        rp[1,0] = rp_t[1,0]

        # 첫 번째 값이 두 번째 값보다 작으면
        if rp_t[0,1] < rp_t[1,1]:
            # +1
            rp[0,1] = 1
            # 아니면 1
            rp[1,1] = -1
        else:
            rp[0,1] = -1
            rp[1,1] = 1

        # 나머지 값은 이전 값과 비교하여 1 -1 설정
        for i in range(2, len(rp_t)):
            rp[i,0] = rp_t[i,0]
            if rp_t[i-1,1] < rp_t[i,1]:
                rp[i,1] = -1
            else:
                rp[i,1] = 1

        # out_arg1 배열 초기화
        out_arg1 = np.zeros(n)

        # rp 배열에 대하여 선형회귀 실행
        for i in range(len(rp)-1):
            # 1차 다항식 회귀 계산식 : 기울기와 y 절편을 반환함
            a = np.polyfit(rp[i:i+2,0], rp[i:i+2,1], 1)

            # y = mx + b 계산식 수행
            for j in range(int(rp[i,0]), int(rp[i+1,0])):
                out_arg1[j] = j*a[0] + a[1]

        # 나머지 범위에 대해서는 out_arg1을 0으로 설정
        for i in range(int(rp[-1,0]), n):
            out_arg1[i] = 0

        # rp 배열의 위치에 해당하는 인덱스에 값을 설정하여 최종 out_arg1 배열 구성
        # 세그먼트에 대한 예측 값을 할당함
        out_arg1[rp[:,0].astype(int)] = rp[:,1]

        # 배열의 마지막 위치값 저장
        out_arg2 = rp[-1,0]

    return out_arg1, out_arg2



# 2. Preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

def FNC_02_Preprocessing(df, T, P):
    # 데이터 로딩
    data = pd.DataFrame()
    scaler = MinMaxScaler(feature_range=(0,1))
    # df['MKTCAP'] = df['MKTCAP'].str.replace(',', '').astype(float)

    # KOSPI KOSDAQ는 시가총액, 나머지는 종가를 기준으로 가격 값을 일정 범위로 스케일링
    # KOSPI KOSDAQ 경우 증자 감자가 때문에 가격이 급격하게 변할 수 있으므로 시가총액을 기준으로 함.
    for i, data in enumerate(df):
        scaler = StandardScaler()
        df['A1'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

        scaler = MinMaxScaler(feature_range=(0, 1))
        df['A2'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

        scaler = MinMaxScaler(feature_range=(-1, 1))
        df['A3'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

        # print(f"Scaling: {i+1} Done.")

    # 주가의 Target으로 사용할 Price_Series_Transform
    # 지그재그형 주가를 T, P를 기준으로 smooting

    # PST 파라매터 설정
    # T=5  # time interval
    # P=1.00  # percentage

    # PST
    for j, data in enumerate(df):
        # nPST=0 means there is no BUY and SELL point by PST
        if df['A2'].size == 0 or len(df['A2']) <= 10:
            # too short time series to calculate pst
            df['PST'] = []
            df['nPST'] = 0
        else:
            df['PST'], df['nPST'] = FNC_Func_PST(T, P, df['A2'].values)

        #print(f"PST: {j+1} Done.")

    # Save
    return df['PST']

import warnings

def trend_gridSearch(df):
  array_T = list(range(2, 11))
  array_P = [0.5 + 0.5 * i for i in range(int(10.00 / 0.5) + 1)]

  max_returns = float('-inf')  # Initialize max_returns to negative infinity
  best_T = None
  best_P = None

  for T in array_T:
      for P in array_P:
          try:
              # 트렌드 만들기
              result = FNC_02_Preprocessing(df, T, P)

              # Backtest--------------------------------------
              # 초기 자본금 설정
              initial_capital = 10000

              # 보유 주식 수와 자본금 추적
              shares_held = 0
              capital = initial_capital
              capital_history = [capital]

              # 매수, 매도, 또는 보유 결정에 따른 자본금 변화 계산
              for i in range(1, len(df)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if result[i] == 1:  # Buy 시그널인 경우
                        shares_to_buy = capital // df['Close'][i]  # 보유 가능한 주식 수 계산
                        shares_held += shares_to_buy

                        price = shares_to_buy * df['Close'][i]
                        commission = price * 0.00015  # 주식 매수수수료

                        capital -= price + commission  # 주식 매수 후 자본금

                    elif result[i] == -1:  # Sell 시그널인 경우
                        if shares_held > 0:  # 주식을 보유하고 있는 경우에만 매도 시그널 처리
                            price = shares_held * df['Close'][i]
                            commission = price * 0.00015 + price * 0.0023  # 주식 매도수수료 : 매도는 세금(0.23%)도 붙음.

                            capital = capital + price - commission  # 보유 주식 매도 후 자본금
                            shares_held = 0  # 보유 주식 수 0으로 초기화

                    capital_history.append(capital + shares_held * df['Close'][i])  # 자본금 변화 추적
              # 수익률 계산
              returns = (capital_history[-1] - initial_capital) / initial_capital * 100

              # print(f"T: {T}, P: {P:.2f}, Returns: {returns}")

              # Update max_returns and corresponding T, P values
              if returns > max_returns:
                  max_returns = returns
                  best_T = T
                  best_P = P

          # 예외 처리
          except (IndexError, ValueError) as e:
              # print(f"{type(e).__name__} occurred. Skipping T: {T}, P: {P:.2f}")

              # Break out of the inner loop when either IndexError or ValueError occurs
              break

      # print(f"Best T: {best_T}, Best P: {best_P:.2f}, Max Returns: {max_returns}")
      print(f"Best T: {best_T}, Best P: {best_P}, Max Returns: {max_returns}")
      return best_T, best_P
