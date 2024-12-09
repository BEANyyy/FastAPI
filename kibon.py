import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import io
import base64


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


# 그래프 그리기
def save_graph(df, index, buy_signals, sell_signals, returns):
    # 매수 시그널과 매도 시그널을 하나의 리스트로 합침
    all_signals = buy_signals + sell_signals
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

    # 매수/매도 시그널 마커에 텍스트 추가
    for date, price, capital, shares in buy_signals:
        ax1.text(date, price * 1.005, f'Close: {price:.2f}\nCapital: {capital:,.0f}\nShares: {shares}',
                 color='green', fontsize=8, ha='center')
    for date, price, capital, shares in sell_signals:
        ax1.text(date, price * 0.995, f'Close: {price:.2f}\nCapital: {capital:,.0f}\nShares: {shares}',
                 color='red', fontsize=8, ha='center')

    # 축 및 제목 설정
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price')
    ax1.legend(loc='upper left')

    # 수익률을 제목에 포함
    plt.title(f'{index} (Returns: {returns:.2f}%)', fontsize=16)

    return fig

def encode_figure_to_base64(fig: Figure) -> str:
    """Matplotlib Figure 객체를 Base64로 인코딩"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    base64_image = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return base64_image




#-------------------pred_signal.py-----------------------------
from TAI.rsi import rsi_test
from TAI.sma import sma_test
from TAI.roc import roc_test
from TAI.dpo import dpo_test
from TAI.stoch import stoch_test
from TAI.macd import merged_macd_test
from TAI.gdc import gdc_test



def save_tai_result(df, stock, model):
    # 시그널 열 저장------------------------------------------------------------------
    # -------INDEX-------
    rsi_result = rsi_test(df, stock, model)
    df['RSI_sig'] = rsi_result[0]

    sma_result = sma_test(df, stock, model)
    df['SMA_sig'] = sma_result[0]

    roc_result = roc_test(df, stock, model)
    df['ROC_sig'] = roc_result[0]

    dpo_result = dpo_test(df, stock, model)
    df['DPO_sig'] = dpo_result[0]

    stoch_result = stoch_test(df, stock, model)
    df['STOCH_sig'] = stoch_result[0]

    macd_result = merged_macd_test(df, stock, model)
    df['MACD_sig'] = macd_result[0]

    gdc_result = gdc_test(df, stock, model)
    df['GDC_sig'] = gdc_result[0]

    print("rsi sell, rsi buy, rsi time, rsi fit : ", rsi_result[1], rsi_result[2], rsi_result[3], rsi_result[4])
    print("sma time, sma fit : ", sma_result[1], sma_result[2])
    print("roc time, roc fit : ", roc_result[1], roc_result[2])
    print("dpo time, dpo fit : ", dpo_result[1], dpo_result[2])
    print("stoch k, stoch d, stoch fit : ", stoch_result[1], stoch_result[2], stoch_result[3])
    print("macd_fastp, macd_slowp, macd_sigp, macd_fit : ", macd_result[1], macd_result[2], macd_result[3],
          macd_result[4])
    print("gdc short, gdc long, gdc fit : ", gdc_result[1], gdc_result[2], gdc_result[3])


    tai_result = {
        "rsi_result" : rsi_result,
        "sma_result": sma_result,
        "roc_result": roc_result,
        "dpo_result": dpo_result,
        "stoch_result": stoch_result,
        "macd_result": macd_result,
        "gdc_result": gdc_result
    }

    return df, tai_result


def print_backtesting(df):
    # 백테스트
    rsi = backtesting(df, 'rsi')
    sma = backtesting(df, 'sma')
    roc = backtesting(df, 'roc')
    dpo = backtesting(df, 'dpo')
    stoch = backtesting(df, 'stoch')
    macd = backtesting(df, 'macd')
    gdc = backtesting(df, 'gdc')

    print("----------------------------------------")
    print("rsi : ", rsi[0])
    print("sma : ", sma[0])
    print("roc : ", roc[0])
    print("dpo : ", dpo[0])
    print("stoch : ", stoch[0])
    print("macd : ", macd[0])
    print("gdc : ", gdc[0])
    print("----------------------------------------")


from ESN.Trend import FNC_02_Preprocessing

def save_trend_result(df):
    # -------TREND-------
    from ESN.Trend import trend_gridSearch

    trend_gridsearch = trend_gridSearch(df)  # 최적의 T, P 값 부터 구하기
    T, P = trend_gridsearch[0], trend_gridsearch[1]
    print("T, P : ", T, P)
    # T, P = 2, 0.5
    df['TREND'] = FNC_02_Preprocessing(df, T, P)
    # 백테스트
    trend = backtesting(df, 'trend')

    print("----------------------------------------")
    print("TREND : ", trend[0])

    return df


from ESN.ESN import ts_train_test, best_fit_esn_model, create_esn_model, making_sig, result_profit
# DB
from db import get_param
def run_model_by_type(df, stock, model):
    # 빈 DataFrame 생성
    result_df = pd.DataFrame(columns=['best_sparsity', 'best_rho', 'best_noise', 'profits'])

    if model == 'GA' or model == 'NO':
        #-------최종 예측 결과--------
        print("ESN 최적화하는 중임")
        all_data_orign_30 = split_data(df)
        train_X, train_Y, test_X, test_Y = ts_train_test(df)

        # 파일명 그리고 최적 파라미터 받아와서 변수에 저장
        max_profit, best_sparsity, best_rho, best_noise = best_fit_esn_model(train_X, train_Y, test_X, test_Y, all_data_orign_30)


    elif model == 'ESN' or model == 'ALL':
        print("ESN 최적화 안 함.")

        parameters = get_param(stock, 'esn')
        print(f"parameters : {parameters}")
        best_sparsity, best_rho, best_noise = parameters[2], parameters[3], parameters[4]
        print(f"best_sparsity, best_rho, best_noise : {best_sparsity}, {best_rho}, {best_noise}")

    all_data_orign_30 = split_data(df)  # 기존 데이터 30% 로드해오기
    train_X, train_Y, test_X, test_Y = ts_train_test(df)  # 트렌드 데이터 불러오기 및 학습-테스트 데이터 나누기

    pred_tot = create_esn_model(best_sparsity, best_rho, best_noise, train_X, train_Y,
                                test_X)  # 최적 파라미터로 모델 학습 및 예측결과 저장
    all_data_orign_30 = making_sig(pred_tot, test_Y, all_data_orign_30)  # 시그널 생성하기
    profits = result_profit(all_data_orign_30, 'pred_sig')



    # --------------- 최근 30% 데이터에 대하여 수익률 추출 ---------------
    # 결과를 DataFrame에 추가
    # 새로운 결과를 딕셔너리로 정의
    new_row = pd.DataFrame([{
        'best_sparsity': best_sparsity,
        'best_rho': best_rho,
        'best_noise': best_noise,
        'profits': profits,
    }])

    # 기존 DataFrame과 새로운 행을 concat으로 합치기
    result_df = pd.concat([result_df, new_row], ignore_index=True)
    print(result_df)

    # 최종 결과 DataFrame 출력
    print("-------------------- 최종 결과 DataFrame에 출력 --------------------")
    print(all_data_orign_30[
              ['Date', 'Close', 'ga_rsi_signal', 'ga_sma_signal', 'ga_roc_signal', 'ga_dpo_signal', 'ga_stoch_signal',
               'ga_macd_signal', 'ga_gdc_signal', 'pred', 'pred_sig']])

    all_data_orign_30.to_csv('result.csv', index=False)  # 필요할 수도 있으니까 csv에 저장함.
    result = backtesting(all_data_orign_30, 'result')
    # show_graph(all_data_orign_30, 'RESULT', result[1], result[2])

    print("수익률 : ", result[0])
    signal = all_data_orign_30['pred_sig'].iloc[-1]  # 오늘의 예측 시그널

    Date = df['Date'].iloc[-1]
    Close = df['Close'].iloc[-1]

    return Date, Close, signal, result_df