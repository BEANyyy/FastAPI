import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# # 주식 데이터를 불러오는 함수
# def load_data(file_name):
#     # file_path = "/content/drive/MyDrive/0_Capstone/data/merged_completed/" + file_name
#     #file_path = "/content/drive/MyDrive/1_Capstone/data/" + file_name
#     #file_path = "/content/drive/MyDrive/" + file_name
#
#
#     # 주식 데이터 불러오기
#     #df = pd.read_csv(file_path)
#     df = yf.download('TSLA', start=start, end=end)
#     df = df.reset_index()
#     df.columns = df.columns.droplevel(1)
#
#     # 날짜 데이터를 datetime 형식으로 변환
#     df['Date'] = pd.to_datetime(df['Date'])
#     # 거꾸로 뒤집기
#     df = df.loc[::-1]
#     # index reset하기 - 기존 index 제거 O
#     df = df.reset_index(drop=True)
#
#
#     df['Date'] = pd.to_datetime(df['Date'])
#
#     # 쉼표 제거하기
#     def remove_commas(value):
#         if isinstance(value, str):
#             return int(value.replace(',', ''))
#         return value
#
#     df = df.applymap(remove_commas)
#
#     return df

# 기존 30% 주식 데이터를 불러오는 함수
def split_data(df):
    #예측 정확도 계산에 이용될 30% 데이터 추출
    N = len(df)
    traintest_cutoff = int(np.ceil(0.7*N))

    all_data_orign_30  = df[traintest_cutoff:]

    return all_data_orign_30


# MDD 계산 함수
def cal_mdd(df):
  #MDD 계산
  # 1년간 영업일은 252일로 잡아서 window = 252로 설정
  window = 252

  # 2. 종가에서 1년기간 단위 최고치 peak를 구함
  peak = df['Close'].rolling(window, min_periods=1).max()

  # 3. 최고치 대비 현재 종가가 얼마나 하락했는지 구함
  drawdown = df['Close']/peak - 1.0

  # 4. drawdown에서 1년기간 단위로 최저치 max_dd를 구한다.
  # max_dd는 마이너스 값이기에 최저치가 MDD가 된다.
  max_dd = drawdown.rolling(window, min_periods=1).min()
  mdd = max_dd.min()

  return mdd


#-------------------------------------Backtesting 함수 정의-----------------------------------------
def backtesting(df, index):
    # 열 이름 정의
    if index == 'trend':
        ga_signal = 'TREND'
    elif index == 'result':
        ga_signal = 'pred_sig'
    else:
        ga_signal = 'ga_' + index + '_signal'

    # 초기 자본금 설정
    initial_capital = 10000

    # 보유 주식 수와 자본금 추적
    shares_held = 0
    capital = initial_capital
    capital_history = [capital]
    buy_signals = []
    sell_signals = []
    capital_changes = []

    # 매수, 매도, 또는 보유 결정에 따른 자본금 변화 계산
    for i in range(1, len(df)):
        if df[ga_signal][i] == 1:  # Buy 시그널인 경우
            shares_to_buy = capital // df['Close'][i]  # 보유 가능한 주식 수 계산
            shares_held += shares_to_buy

            price = shares_to_buy * df['Close'][i]
            commission = price * 0.00015  # 주식 매수수수료

            capital -= price + commission # 주식 매수 후 자본금


            # 그래프에 표시하기 위한 배열에 항목 추가
            buy_signals.append((df['Date'][i], df['Close'][i], capital, shares_held))  # 매수 시그널 저장

        elif df[ga_signal][i] == -1:  # Sell 시그널인 경우
            if shares_held > 0:  # 주식을 보유하고 있는 경우에만 매도 시그널 처리
                price = shares_held * df['Close'][i]
                commission = price * 0.00015 + price * 0.0023  # 주식 매도수수료 : 매도는 세금(0.23%)도 붙음.

                capital = capital + price - commission  # 보유 주식 매도 후 자본금
                shares_held = 0 # 보유 주식 수 0으로 초기화


                # 그래프에 표시하기 위한 배열에 항목 추가
                sell_signals.append((df['Date'][i], df['Close'][i], capital, shares_held))  # 매도 시그널 저장

        # 자본금 변화 추적
        capital_history.append(capital)
        capital_changes.append((df['Date'][i], capital))

    # 수익률 계산
    returns = (capital_history[-1] - initial_capital) / initial_capital * 100

    # print(capital_history)

    return returns, buy_signals, sell_signals

# -------------------------------------Backtesting 함수 수정 : 한 번에 한 주씩 / 가진 주식도 자본에 포함하여 수익률 계산 -----------------------------------------
def new_backtesting(df, index):

    # 열 이름 정의
    if index == 'trend':
        ga_signal = 'TREND'
    elif index == 'result':
        ga_signal = 'pred_sig'
    else:
        ga_signal = 'ga_' + index + '_signal'

    # 초기 자본금 설정
    initial_capital = 1000

    # 보유 주식 수와 자본금 추적
    shares_held = 0
    capital = initial_capital
    capital_history = [capital]
    buy_signals = []
    sell_signals = []
    capital_changes = []

    # 매수, 매도, 또는 보유 결정에 따른 자본금 변화 계산
    for i in range(1, len(df)):
        if df[ga_signal][i] == 1:  # Buy 시그널인 경우
            if capital >= df['Close'][i]:
                # shares_to_buy = capital // df['Close'][i]  # 보유 가능한 주식 수 계산
                shares_to_buy = 1  # 1주식 사는 것으로 수정

                shares_held += shares_to_buy

                price = shares_to_buy * df['Close'][i]
                commission = price * 0.00015  # 주식 매수수수료

                capital -= price + commission  # 주식 매수 후 자본금

                # 그래프에 표시하기 위한 배열에 항목 추가
                buy_signals.append((df['Date'][i], df['Close'][i], capital, shares_held))  # 매수 시그널 저장

        elif df[ga_signal][i] == -1:  # Sell 시그널인 경우
            if shares_held > 0:  # 주식을 보유하고 있는 경우에만 매도 시그널 처리
                price = shares_held * df['Close'][i]
                commission = price * 0.00015 + price * 0.0023  # 주식 매도수수료 : 매도는 세금(0.23%)도 붙음.

                capital = capital + price - commission  # 보유 주식 매도 후 자본금
                shares_held = 0  # 보유 주식 수 0으로 초기화

                # 그래프에 표시하기 위한 배열에 항목 추가
                sell_signals.append((df['Date'][i], df['Close'][i], capital, shares_held))  # 매도 시그널 저장

        # ==========주식까지 내 자본이 되도록 코드 수정===========
        # 현재 보유 주식 평가 금액 계산
        stock_value = shares_held * df['Close'][i]

        # 총 자본 (현금 + 주식 평가 금액)
        total_capital = capital + stock_value

        # 자본금 변화 추적
        # capital_history.append(capital)
        capital_history.append(total_capital)
        capital_changes.append((df['Date'][i], capital))

    # 수익률 계산
    returns = (capital_history[-1] - initial_capital) / initial_capital * 100

    # print(capital_history)

    return returns, buy_signals, sell_signals


# 그래프 그리는 함수
# 그래프 그리기
def show_graph(df, index, buy_signals, sell_signals):
    # 매수 시그널과 매도 시그널을 하나의 리스트로 합침
    all_signals = buy_signals + sell_signals
    # all_signals.sort(key=lambda x: x[0])  # 날짜 기준으로 정렬
    all_signals.sort(key=lambda x: pd.Timestamp(x[0]))  # datetime.date -> Timestamp 변환

    # 그래프 그리기
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # 종가 가격 그래프
    ax1.plot(df['Date'], df['Close'], label='Close Price', color='blue', linewidth=1.5)

    # 매수, 매도 시그널 (초록 삼각형과 빨간 삼각형) 표시
    ax1.scatter([date for date, price, capital, shares in buy_signals],
                [price for date, price, capital, shares in buy_signals],
                marker='^', color='green', s=100, label='Buy Signal')
    ax1.scatter([date for date, price, capital, shares in sell_signals],
                [price for date, price, capital, shares in sell_signals],
                marker='v', color='red', s=100, label='Sell Signal')

    # 시그널들을 선으로 연결
    signal_dates = [date for date, price, capital, shares in all_signals]
    signal_prices = [price for date, price, capital, shares in all_signals]
    ax1.plot(signal_dates, signal_prices, color='purple', linestyle='-', linewidth=1, label='Signal Line')

    # 자본금, 가격, 보유 주식 수 표시 (마커 위에만 표시)
    for date, price, capital, shares in buy_signals:  # 매수 시그널 마커 위에 자본금과 가격, 주식 수 표시
        ax1.text(date, price * 1.005, f'Close: {price:.2f}\nCapital: {capital:,.0f}\nShares: {shares}',
                 color='green', fontsize=8, ha='center')  # 가격, 자본금, 보유 주식 수 표시
    for date, price, capital, shares in sell_signals:  # 매도 시그널 마커 아래에 자본금과 가격, 주식 수 표시
        ax1.text(date, price * 0.995, f'Close: {price:.2f}\nCapital: {capital:,.0f}\nShares: {shares}',
                 color='red', fontsize=8, ha='center')  # 가격, 자본금, 보유 주식 수 표시

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price')

    # legend는 한 번만 설정하여 중복 방지
    ax1.legend(loc='upper left')

    plt.title(index)
    plt.show()

# 'A'로 값을 가져오는 함수
def get_parameters(key, parameters):
    return parameters.get(key, "Key not found")