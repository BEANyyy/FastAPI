import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
from dateutil.relativedelta import relativedelta

# TAI
from kibon import show_graph, save_tai_result, save_trend_result, print_backtesting, run_model_by_type



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


def pred_signal(i, j, stock, model):
    df, end = bring_data(stock, j)
    print(f"최적화 타입 : {model}만 최적화된 상태에서 진행 중")
    print("====================", j, "개 남음 (", i+1, "/3)  ====================")
    df = save_tai_result(df, stock, model)[0]
    # print_backtesting(df)

    df = save_trend_result(df)
    Date, Close, signal, result_df = run_model_by_type(df, stock, model)

    return Date, Close, signal, result_df


def update_param_pred_signal(i, j, stock, model):
    df, end = bring_data(stock, j)
    print(f"최적화 타입 : {model}만 최적화된 상태에서 진행 중")
    print("====================", j, "개 남음 (", i+1, "/3)  ====================")
    df, tai_result = save_tai_result(df, stock, model)
    # print_backtesting(df)

    rsi_result = tai_result["rsi_result"]
    sma_result = tai_result["sma_result"]
    roc_result = tai_result["roc_result"]
    dpo_result = tai_result["dpo_result"]
    stoch_result = tai_result["stoch_result"]
    macd_result = tai_result["macd_result"]
    gdc_result = tai_result["gdc_result"]

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
    sma_df = pd.concat([sma_df, sma_new_row], ignore_index=True)
    roc_df = pd.concat([roc_df, roc_new_row], ignore_index=True)
    dpo_df = pd.concat([dpo_df, dpo_new_row], ignore_index=True)
    stoch_df = pd.concat([stoch_df, stoch_new_row], ignore_index=True)
    macd_df = (pd.concat([macd_df, macd_new_row], ignore_index=True))
    gdc_df = pd.concat([gdc_df, gdc_new_row], ignore_index=True)

    df = save_trend_result(df)
    Date, Close, signal, result_df = run_model_by_type(df, stock, model)

    return Date, Close, signal, result_df, rsi_df, sma_df, roc_df, dpo_df, stoch_df, macd_df, gdc_df




def realtime_pred_signal(i, j, stock, model, close):
    print("pred_signal_realtime")
    df, end = bring_data(stock, j)
    print(f"최적화 타입 : {model}만 최적화된 상태에서 진행 중")

    date = pd.Timestamp(end, tz='UTC')
    print(date)

    # 'Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'
    new_data = {'Date': date, 'Close': close}
    new_row = pd.DataFrame([new_data])

    # 새로운 데이터 추가
    df = pd.concat([df, new_row], ignore_index=True)
    print(df[['Date', 'Close']].tail(5))

    print("====================", j, "개 남음 (", i+1, "/3)  ====================")
    df = save_tai_result(df, stock, model)[0]
    # print_backtesting(df)

    df = save_trend_result(df)
    Date, Close, signal, result_df = run_model_by_type(df, stock, model)

    return Date, Close, signal, result_df
