import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
from dateutil.relativedelta import relativedelta

# TAI
from kibon import split_data, backtesting, show_graph
from TAI.rsi import rsi_test
from TAI.sma import sma_test
from TAI.roc import roc_test
from TAI.dpo import dpo_test
from TAI.stoch import stoch_test
from TAI.macd import merged_macd_test
from TAI.gdc import gdc_test

from ESN.ESN import ts_train_test, create_esn_model, making_sig, result_profit, best_fit_esn_model
from ESN.Trend import FNC_02_Preprocessing

# DB
from db import get_param

def bring_data(stock, j):
    # 날짜 범위 설정
    # 미국 뉴욕 시간대 설정
    us_time_zone = pytz.timezone('America/New_York')

    print(f"현재 최적화하는 주식 :  {stock}")
    # 현재 시간(미국 뉴욕 시간대 기준) 가져오기
    today = datetime.now(us_time_zone).date()
    # end = today + relativedelta(days=1)
    end = today - relativedelta(days=j)
    print("미국 뉴욕 시간 기준 현재 날짜:", end)

    # 6개월 전 날짜 계산
    start = end - relativedelta(months=12)
    print("미국 뉴욕 시간 기준 1년 전 날짜:", start)

    # MSFT 데이터 다운로드
    df = yf.download(stock, start=start, end=end)
    df = df.reset_index()
    df.columns = df.columns.droplevel(1)

    return df, end



def pred_signal(i, j, stock, type):
    df, end = bring_data(stock, j)
    print(f"최적화 타입 : {type}만 최적화된 상태에서 진행 중")


    print("====================", j, "개 남음 (", i+1, "/3)  ====================")
    # 시그널 열 저장------------------------------------------------------------------
    # -------INDEX-------
    rsi_result = rsi_test(df, stock, type)
    df['RSI_sig'] = rsi_result[0]

    sma_result = sma_test(df, stock, type)
    df['SMA_sig'] = sma_result[0]

    roc_result = roc_test(df, stock, type)
    df['ROC_sig'] = roc_result[0]

    dpo_result = dpo_test(df, stock, type)
    df['DPO_sig'] = dpo_result[0]

    stoch_result = stoch_test(df, stock, type)
    df['STOCH_sig'] = stoch_result[0]

    macd_result = merged_macd_test(df, stock, type)
    df['MACD_sig'] = macd_result[0]

    gdc_result = gdc_test(df, stock, type)
    df['GDC_sig'] = gdc_result[0]


    print("rsi sell, rsi buy, rsi time, rsi fit : ", rsi_result[1], rsi_result[2], rsi_result[3], rsi_result[4])
    print("sma time, sma fit : ", sma_result[1], sma_result[2])
    print("roc time, roc fit : ", roc_result[1], roc_result[2])
    print("dpo time, dpo fit : ", dpo_result[1], dpo_result[2])
    print("stoch k, stoch d, stoch fit : ", stoch_result[1], stoch_result[2], stoch_result[3])
    print("macd_fastp, macd_slowp, macd_sigp, macd_fit : ", macd_result[1], macd_result[2], macd_result[3], macd_result[4])
    print("gdc short, gdc long, gdc fit : ", gdc_result[1], gdc_result[2], gdc_result[3])
    #
    # # 백테스트
    # rsi = backtesting(df, 'rsi')
    # sma = backtesting(df, 'sma')
    # roc = backtesting(df, 'roc')
    # dpo = backtesting(df, 'dpo')
    # stoch = backtesting(df, 'stoch')
    # macd = backtesting(df, 'macd')
    # gdc = backtesting(df, 'gdc')
    #
    # print("----------------------------------------")
    # print("rsi : ", rsi[0])
    # print("sma : ", sma[0])
    # print("roc : ", roc[0])
    # print("dpo : ", dpo[0])
    # print("stoch : ", stoch[0])
    # print("macd : ", macd[0])
    # print("gdc : ", gdc[0])
    # print("----------------------------------------")



    # -------TREND-------
    from ESN.Trend import trend_gridSearch
    trend_gridSearch = trend_gridSearch(df)  # 최적의 T, P 값 부터 구하기
    T, P = trend_gridSearch[0], trend_gridSearch[1]
    print("T, P : ", T, P)
    # T, P = 2, 0.5
    df['TREND'] = FNC_02_Preprocessing(df, T, P)
    # 백테스트
    trend = backtesting(df, 'trend')

    print("----------------------------------------")
    print("TREND : ", trend[0])

    # 빈 DataFrame 생성
    result_df = pd.DataFrame(columns=['best_sparsity', 'best_rho', 'best_noise', 'profits'])


    if type == 'GA' or type == 'NO':
        #-------최종 예측 결과--------
        print("ESN 최적화하는 중임")
        all_data_orign_30 = split_data(df)
        train_X, train_Y, test_X, test_Y = ts_train_test(df)

        # 파일명 그리고 최적 파라미터 받아와서 변수에 저장
        max_profit, best_sparsity, best_rho, best_noise = best_fit_esn_model(train_X, train_Y, test_X, test_Y, all_data_orign_30)


    elif type == 'ESN' or type == 'ALL':
        print("ESN 최적화 안 함.")

        parameters = get_param(stock, 'esn')
        print(f"parameters : {parameters}")
        best_sparsity, best_rho, best_noise = parameters[2], parameters[3], parameters[4]
        print(f"best_sparsity, best_rho, best_noise : {best_sparsity}, {best_rho}, {best_noise}")


    all_data_orign_30 = split_data(df)  # 기존 데이터 30% 로드해오기
    train_X, train_Y, test_X, test_Y = ts_train_test(df)  # 트렌드 데이터 불러오기 및 학습-테스트 데이터 나누기

    pred_tot = create_esn_model(best_sparsity, best_rho, best_noise, train_X, train_Y, test_X)  # 최적 파라미터로 모델 학습 및 예측결과 저장
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
        # 'matching_percentage': matching_percentage
    }])

    # 기존 DataFrame과 새로운 행을 concat으로 합치기
    result_df = pd.concat([result_df, new_row], ignore_index=True)
    print(result_df)



    # 최종 결과 DataFrame 출력
    print("-------------------- 최종 결과 DataFrame에 출력 --------------------")
    print(all_data_orign_30[['Date', 'Close', 'ga_rsi_signal', 'ga_sma_signal', 'ga_roc_signal', 'ga_dpo_signal', 'ga_stoch_signal', 'ga_macd_signal', 'ga_gdc_signal',  'pred', 'pred_sig']])

    all_data_orign_30.to_csv('result.csv', index=False) # 필요할 수도 있으니까 csv에 저장함.
    result = backtesting(all_data_orign_30, 'result')
    # show_graph(all_data_orign_30, 'RESULT', result[1], result[2])


    print("수익률 : ", result[0])
    signal = all_data_orign_30['pred_sig'].iloc[-1]  # 오늘의 예측 시그널

    Date = df['Date'].iloc[-1]
    Close = df['Close'].iloc[-1]


    return Date, Close, signal, result_df


def update_param_pred_signal(i, j, stock, type):
    df, end = bring_data(stock, j)
    print(f"최적화 타입 : {type}만 최적화된 상태에서 진행 중")

    print("====================", j, "개 남음 (", i+1, "/3)  ====================")

    # 시그널 열 저장------------------------------------------------------------------
    # -------INDEX-------
    rsi_result = rsi_test(df, stock, type)
    df['RSI_sig'] = rsi_result[0]

    sma_result = sma_test(df, stock, type)
    df['SMA_sig'] = sma_result[0]

    roc_result = roc_test(df, stock, type)
    df['ROC_sig'] = roc_result[0]

    dpo_result = dpo_test(df, stock, type)
    df['DPO_sig'] = dpo_result[0]

    stoch_result = stoch_test(df, stock, type)
    df['STOCH_sig'] = stoch_result[0]

    macd_result = merged_macd_test(df, stock, type)
    df['MACD_sig'] = macd_result[0]

    gdc_result = gdc_test(df, stock, type)
    df['GDC_sig'] = gdc_result[0]



    # ---------- 결과를 저장할 df 생성 ------------
    rsi_df = pd.DataFrame(columns=['rsi_sell', 'rsi_buy', 'rsi_time', 'fit'])
    sma_df = pd.DataFrame(columns=['sma_time', 'fit'])
    roc_df = pd.DataFrame(columns=['roc_time', 'fit'])
    dpo_df = pd.DataFrame(columns=['dpo_time', 'fit'])
    stoch_df = pd.DataFrame(columns=['stoch_k', 'stoch_d', 'fit'])
    macd_df = pd.DataFrame(columns=['macd_fastp', 'macd_slowp', 'macd_sigp', 'fit'])
    gdc_df = pd.DataFrame(columns=['gdc_short', 'gdc_long', 'fit'])

    # --------------- df에 결과 저장 ---------------
    rsi_sell, rsi_buy, rsi_time, rsi_fit = rsi_result[1], rsi_result[2], rsi_result[3], rsi_result[4]
    sma_time, sma_fit = sma_result[1], sma_result[2]
    roc_time , roc_fit = roc_result[1], roc_result[2]
    dpo_time, dpo_fit = dpo_result[1], dpo_result[2]
    stoch_k, stoch_d, stoch_fit = stoch_result[1], stoch_result[2], stoch_result[3]
    macd_fastp, macd_slowp, macd_sigp, macd_fit = macd_result[1], macd_result[2], macd_result[3], macd_result[4]
    gdc_short, gdc_long, gdc_fit= gdc_result[1], gdc_result[2], gdc_result[3]


    # 결과를 DataFrame에 추가
    # 새로운 결과를 딕셔너리로 정의
    rsi_new_row = pd.DataFrame([{
        'rsi_sell': rsi_sell,
        'rsi_buy': rsi_buy,
        'rsi_time': rsi_time,
        'fit': rsi_fit,
    }])
    sma_new_row = pd.DataFrame([{
        'sma_time': sma_time,
        'fit': sma_fit,
    }])
    roc_new_row = pd.DataFrame([{
        'roc_time': roc_time,
        'fit': roc_fit,
    }])
    dpo_new_row = pd.DataFrame([{
        'dpo_time': dpo_time,
        'fit': dpo_fit,
    }])
    stoch_new_row = pd.DataFrame([{
        'stoch_k': stoch_k,
        'stoch_d' : stoch_d,
        'fit': stoch_fit,
    }])
    macd_new_row = pd.DataFrame([{
        'macd_fastp': macd_fastp,
        'macd_slowp': macd_slowp,
        'macd_sigp': macd_sigp,
        'fit': macd_fit,
    }])
    gdc_new_row = pd.DataFrame([{
        'gdc_short': gdc_short,
        'gdc_long' : gdc_long,
        'fit': gdc_fit,
    }])

    # 기존 DataFrame과 새로운 행을 concat으로 합치기
    rsi_df = pd.concat([rsi_df, rsi_new_row], ignore_index=True)
    # print(rsi_df)

    sma_df = pd.concat([sma_df, sma_new_row], ignore_index=True)
    # print(sma_df)

    roc_df = pd.concat([roc_df, roc_new_row], ignore_index=True)
    # print(roc_df)

    dpo_df = pd.concat([dpo_df, dpo_new_row], ignore_index=True)
    # print(dpo_df)

    stoch_df = pd.concat([stoch_df, stoch_new_row], ignore_index=True)
    # print(stoch_df)

    macd_df = (pd.concat([macd_df, macd_new_row], ignore_index=True))
    # print(macd_df)

    gdc_df = pd.concat([gdc_df, gdc_new_row], ignore_index=True)
    # print(gdc_df)


    print("rsi sell, rsi buy, rsi time, rsi fit : ", rsi_result[1], rsi_result[2], rsi_result[3], rsi_result[4])
    print("sma time, sma fit : ", sma_result[1], sma_result[2])
    print("roc time, roc fit : ", roc_result[1], roc_result[2])
    print("dpo time, dpo fit : ", dpo_result[1], dpo_result[2])
    print("stoch k, stoch d, stoch fit : ", stoch_result[1], stoch_result[2], stoch_result[3])
    print("macd_fastp, macd_slowp, macd_sigp, macd_fit : ", macd_result[1], macd_result[2], macd_result[3], macd_result[4])
    print("gdc short, gdc long, gdc fit : ", gdc_result[1], gdc_result[2], gdc_result[3])
    #
    # # 백테스트
    # rsi = backtesting(df, 'rsi')
    # sma = backtesting(df, 'sma')
    # roc = backtesting(df, 'roc')
    # dpo = backtesting(df, 'dpo')
    # stoch = backtesting(df, 'stoch')
    # macd = backtesting(df, 'macd')
    # gdc = backtesting(df, 'gdc')
    #
    # print("----------------------------------------")
    # print("rsi : ", rsi[0])
    # print("sma : ", sma[0])
    # print("roc : ", roc[0])
    # print("dpo : ", dpo[0])
    # print("stoch : ", stoch[0])
    # print("macd : ", macd[0])
    # print("gdc : ", gdc[0])
    # print("----------------------------------------")



    # -------TREND-------
    from ESN.Trend import trend_gridSearch
    trend_gridSearch = trend_gridSearch(df)  # 최적의 T, P 값 부터 구하기
    T, P = trend_gridSearch[0], trend_gridSearch[1]
    print("T, P : ", T, P)
    # T, P = 2, 0.5
    df['TREND'] = FNC_02_Preprocessing(df, T, P)
    # 백테스트
    trend = backtesting(df, 'trend')

    print("----------------------------------------")
    print("TREND : ", trend[0])

    # 빈 DataFrame 생성
    result_df = pd.DataFrame(columns=['best_sparsity', 'best_rho', 'best_noise', 'profits'])


    if type == 'GA' or type == 'NO':
        #-------최종 예측 결과--------
        print("ESN 최적화하는 중임")
        all_data_orign_30 = split_data(df)
        train_X, train_Y, test_X, test_Y = ts_train_test(df)

        # 파일명 그리고 최적 파라미터 받아와서 변수에 저장
        max_profit, best_sparsity, best_rho, best_noise = best_fit_esn_model(train_X, train_Y, test_X, test_Y, all_data_orign_30)


    elif type == 'ESN' or type == 'ALL':
        print("ESN 최적화 안 함.")

        parameters = get_param(stock, 'esn')
        print(f"parameters : {parameters}")
        best_sparsity, best_rho, best_noise = parameters[2], parameters[3], parameters[4]
        print(f"best_sparsity, best_rho, best_noise : {best_sparsity}, {best_rho}, {best_noise}")


    all_data_orign_30 = split_data(df)  # 기존 데이터 30% 로드해오기
    train_X, train_Y, test_X, test_Y = ts_train_test(df)  # 트렌드 데이터 불러오기 및 학습-테스트 데이터 나누기

    pred_tot = create_esn_model(best_sparsity, best_rho, best_noise, train_X, train_Y, test_X)  # 최적 파라미터로 모델 학습 및 예측결과 저장
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
        # 'matching_percentage': matching_percentage
    }])

    # 기존 DataFrame과 새로운 행을 concat으로 합치기
    result_df = pd.concat([result_df, new_row], ignore_index=True)
    print(result_df)



    # 최종 결과 DataFrame 출력
    print("-------------------- 최종 결과 DataFrame에 출력 --------------------")
    print(all_data_orign_30[['Date', 'Close', 'ga_rsi_signal', 'ga_sma_signal', 'ga_roc_signal', 'ga_dpo_signal', 'ga_stoch_signal', 'ga_macd_signal', 'ga_gdc_signal',  'pred', 'pred_sig']])

    all_data_orign_30.to_csv('result.csv', index=False) # 필요할 수도 있으니까 csv에 저장함.
    result = backtesting(all_data_orign_30, 'result')
    # show_graph(all_data_orign_30, 'RESULT', result[1], result[2])


    print("수익률 : ", result[0])
    signal = all_data_orign_30['pred_sig'].iloc[-1]  # 오늘의 예측 시그널

    Date = df['Date'].iloc[-1]
    Close = df['Close'].iloc[-1]


    return Date, Close, signal, result_df, rsi_df, sma_df, roc_df, dpo_df, stoch_df, macd_df, gdc_df




def realtime_pred_signal(i, j, stock, type, close):
    print("pred_signal_realtime")
    df, end = bring_data(stock, j)
    print(f"최적화 타입 : {type}만 최적화된 상태에서 진행 중")

    date = pd.Timestamp(end, tz='UTC')
    print(date)

    # 'Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'
    new_data = {'Date': date, 'Close': close}
    new_row = pd.DataFrame([new_data])

    # 새로운 데이터 추가
    df = pd.concat([df, new_row], ignore_index=True)
    print(df[['Date', 'Close']].tail(5))

    print("====================", j, "개 남음 (", i+1, "/3)  ====================")

    # 시그널 열 저장------------------------------------------------------------------
    # -------INDEX-------
    rsi_result = rsi_test(df, stock, type)
    df['RSI_sig'] = rsi_result[0]

    sma_result = sma_test(df, stock, type)
    df['SMA_sig'] = sma_result[0]

    roc_result = roc_test(df, stock, type)
    df['ROC_sig'] = roc_result[0]

    dpo_result = dpo_test(df, stock, type)
    df['DPO_sig'] = dpo_result[0]

    stoch_result = stoch_test(df, stock, type)
    df['STOCH_sig'] = stoch_result[0]

    macd_result = merged_macd_test(df, stock, type)
    df['MACD_sig'] = macd_result[0]

    gdc_result = gdc_test(df, stock, type)
    df['GDC_sig'] = gdc_result[0]


    print("rsi sell, rsi buy, rsi time, rsi fit : ", rsi_result[1], rsi_result[2], rsi_result[3], rsi_result[4])
    print("sma time, sma fit : ", sma_result[1], sma_result[2])
    print("roc time, roc fit : ", roc_result[1], roc_result[2])
    print("dpo time, dpo fit : ", dpo_result[1], dpo_result[2])
    print("stoch k, stoch d, stoch fit : ", stoch_result[1], stoch_result[2], stoch_result[3])
    print("macd_fastp, macd_slowp, macd_sigp, macd_fit : ", macd_result[1], macd_result[2], macd_result[3], macd_result[4])
    print("gdc short, gdc long, gdc fit : ", gdc_result[1], gdc_result[2], gdc_result[3])


    # -------TREND-------
    from ESN.Trend import trend_gridSearch
    trend_gridSearch = trend_gridSearch(df)  # 최적의 T, P 값 부터 구하기
    T, P = trend_gridSearch[0], trend_gridSearch[1]
    print("T, P : ", T, P)
    # T, P = 2, 0.5
    df['TREND'] = FNC_02_Preprocessing(df, T, P)
    # 백테스트
    trend = backtesting(df, 'trend')

    print("----------------------------------------")
    print("TREND : ", trend[0])

    # 빈 DataFrame 생성
    result_df = pd.DataFrame(columns=['best_sparsity', 'best_rho', 'best_noise', 'profits'])


    if type == 'GA' or type == 'NO':
        #-------최종 예측 결과--------
        print("ESN 최적화하는 중임")
        all_data_orign_30 = split_data(df)
        train_X, train_Y, test_X, test_Y = ts_train_test(df)

        # 파일명 그리고 최적 파라미터 받아와서 변수에 저장
        max_profit, best_sparsity, best_rho, best_noise = best_fit_esn_model(train_X, train_Y, test_X, test_Y, all_data_orign_30)


    elif type == 'ESN' or type == 'ALL':
        print("ESN 최적화 안 함.")

        parameters = get_param(stock, 'esn')
        print(f"parameters : {parameters}")
        best_sparsity, best_rho, best_noise = parameters[2], parameters[3], parameters[4]
        print(f"best_sparsity, best_rho, best_noise : {best_sparsity}, {best_rho}, {best_noise}")


    all_data_orign_30 = split_data(df)  # 기존 데이터 30% 로드해오기
    train_X, train_Y, test_X, test_Y = ts_train_test(df)  # 트렌드 데이터 불러오기 및 학습-테스트 데이터 나누기

    pred_tot = create_esn_model(best_sparsity, best_rho, best_noise, train_X, train_Y, test_X)  # 최적 파라미터로 모델 학습 및 예측결과 저장
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
        # 'matching_percentage': matching_percentage
    }])

    # 기존 DataFrame과 새로운 행을 concat으로 합치기
    result_df = pd.concat([result_df, new_row], ignore_index=True)
    print(result_df)



    # 최종 결과 DataFrame 출력
    print("-------------------- 최종 결과 DataFrame에 출력 --------------------")
    print(all_data_orign_30[['Date', 'Close', 'ga_rsi_signal', 'ga_sma_signal', 'ga_roc_signal', 'ga_dpo_signal', 'ga_stoch_signal', 'ga_macd_signal', 'ga_gdc_signal',  'pred', 'pred_sig']])

    all_data_orign_30.to_csv('result.csv', index=False) # 필요할 수도 있으니까 csv에 저장함.
    result = backtesting(all_data_orign_30, 'result')
    # show_graph(all_data_orign_30, 'RESULT', result[1], result[2])


    print("수익률 : ", result[0])
    signal = all_data_orign_30['pred_sig'].iloc[-1]  # 오늘의 예측 시그널

    Date = df['Date'].iloc[-1]
    Close = df['Close'].iloc[-1]


    return Date, Close, signal, result_df
