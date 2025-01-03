import logging
import pandas as pd

# DB
from db import connect_to_db, insert_data

from make_signal import update_param_pred_signal

def update_param(stock, type, back_date, today_date, times):
    # 빈 DataFrame 생성
    test_df = pd.DataFrame(columns=['Date', 'Close', 'pred_sig'])

    for j in range(back_date, today_date, -1):
        signal_result = [0, 0, 0]  # -1, 1, 0

        rsi_data = None
        sma_data = None
        roc_data = None
        dpo_data = None
        stoch_data = None
        macd_data = None
        gdc_data = None
        esn_data = None

        # 빈 DataFrame 생성
        rsi_df = pd.DataFrame(columns=['rsi_sell', 'rsi_buy', 'rsi_time', 'fit'])
        sma_df = pd.DataFrame(columns=['sma_time', 'fit'])
        roc_df = pd.DataFrame(columns=['roc_time', 'fit'])
        dpo_df = pd.DataFrame(columns=['dpo_time', 'fit'])
        stoch_df = pd.DataFrame(columns=['stoch_k', 'stoch_d', 'fit'])
        macd_df = pd.DataFrame(columns=['macd_fastp', 'macd_slowp', 'macd_sigp', 'fit'])
        gdc_df = pd.DataFrame(columns=['gdc_short', 'gdc_long', 'fit'])

        # 빈 DataFrame 생성
        esn_df = pd.DataFrame(columns=['best_sparsity', 'best_rho', 'best_noise', 'profits'])


        for i in range(times):
            Date, Close, signal, esn_param, rsi_param, sma_param, roc_param, dpo_param, stoch_param, macd_param, gdc_param = update_param_pred_signal(i, j, stock, type)

            rsi_df = pd.concat([rsi_df, rsi_param], ignore_index=True)
            sma_df = pd.concat([sma_df, sma_param], ignore_index=True)
            roc_df = pd.concat([roc_df, roc_param], ignore_index=True)
            dpo_df = pd.concat([dpo_df, dpo_param], ignore_index=True)
            stoch_df = pd.concat([stoch_df, stoch_param], ignore_index=True)
            macd_df = pd.concat([macd_df, macd_param], ignore_index=True)
            gdc_df = pd.concat([gdc_df, gdc_param], ignore_index=True)

            # A열에서 가장 큰 값이 있는 행 출력
            max_rsi = rsi_df[rsi_df['fit'] == rsi_df['fit'].max()]
            max_sma = sma_df[sma_df['fit'] == sma_df['fit'].max()]
            max_roc = roc_df[roc_df['fit'] == roc_df['fit'].max()]
            max_dpo = dpo_df[dpo_df['fit'] == dpo_df['fit'].max()]
            max_stoch = stoch_df[stoch_df['fit'] == stoch_df['fit'].max()]
            max_macd = macd_df[macd_df['fit'] == macd_df['fit'].max()]
            max_gdc = gdc_df[gdc_df['fit'] == gdc_df['fit'].max()]

            rsi_data = max_rsi.head(1).iloc[:, :-1]    # db에 넣어야 되니까 fit은 뺌.
            sma_data = max_sma.head(1).iloc[:, :-1]
            roc_data = max_roc.head(1).iloc[:, :-1]
            dpo_data = max_dpo.head(1).iloc[:, :-1]
            stoch_data = max_stoch.head(1).iloc[:, :-1]
            macd_data = max_macd.head(1).iloc[:, :-1]
            gdc_data = max_gdc.head(1).iloc[:, :-1]

            # 추출한 esn param 결과를 esn_df에 합치기 : profit 비교하여 제일 큰 param을 남기기 위함
            esn_df = pd.concat([esn_df, esn_param], ignore_index=True)
            print(esn_df)
            # A열에서 가장 큰 값이 있는 행 출력
            esn_data = esn_df[esn_df['profits'] == esn_df['profits'].max()].iloc[:, :-1]



            if signal == -1:
                signal_result[0] += 1
            elif signal == 1:
                signal_result[1] += 1
            else:
                signal_result[2] += 1


        print("=========tai_max_row=========")
        print(rsi_data)
        print(sma_data)
        print(roc_data)
        print(dpo_data)
        print(stoch_data)
        print(macd_data)
        print(gdc_data)

        print("=========esn_max_row=========")
        print(f"best_sparsity, best_rho, best_noise : {esn_data[['best_sparsity', 'best_rho', 'best_noise']]}")


        # ===========추출한 param을 db에 저장=============
        print("__name__ :", __name__)   
        if __name__ == "__main__":  # 그냥 여기서 실행 시
        # if __name__ == "stock":     # .bat 파일 실행 시
            # 연결 정보 설정
            host = "localhost"  # MySQL 서버 호스트 (로컬의 경우 localhost)
            user = "root"  # MySQL 사용자 이름
            password = "1234"  # MySQL 비밀번호
            database = "stock_db"  # 데이터베이스 이름

            # 데이터베이스 연결
            connection = connect_to_db(host, user, password, database)

            print(Date)
            if connection:
                # ---------데이터 삽입--------
                tai_list =[('rsi', rsi_data), ('sma', sma_data), ('roc', roc_data), ('dpo', dpo_data), ('stoch', stoch_data), ('macd', macd_data), ('gdc', gdc_data), ('esn', esn_data)]

                for tai in tai_list:
                    # {tai}_data에서 값 추출 및 변환
                    print(tai[0])
                    tai_dict = {}
                    for _, row in tai[1].iterrows():
                        tai_dict = row.to_dict()  # 행을 딕셔너리로 변환

                    # 기존 데이터와 병합
                    data = {
                        'stock_id': stock,  # 추가할 고유 키
                        **tai_dict,  # rsi_data의 컬럼과 값을 병합
                        'update_date': Date  # 추가할 날짜
                    }

                    print(data)

                    # # 함수 호출
                    insert_data(connection, tai[0], data)


                    print("새로운 param 삽입 완료 ! !! ! !")
                connection.close()

        print(signal_result)
        if signal_result.index(max(signal_result)) == 0: # -1일 때 (인덱스 : 0)
            final_signal = -1
            print("오늘의 거래 제안 : 파세요")
        elif signal_result.index(max(signal_result)) == 1: # 1일 때 (인덱스 : 1)
            final_signal = 1
            print("오늘의 거래 제안 : 사세요")
        else:   # 0일 때 (인덱스 : 2)
            final_signal = 0
            print("오늘의 거래 제안 : 유지")


        new_row = pd.DataFrame([{
            'Date': Date,
            'Close': Close,
            'pred_sig': final_signal,
            # 'matching_percentage': matching_percentage
        }])

        # 기존 DataFrame과 새로운 행을 concat으로 합치기
        test_df = pd.concat([test_df, new_row], ignore_index=True)

        print("-------test_df------")
        print(test_df)


# 로그 설정
logging.basicConfig(
    filename='script_log.log',  # 로그 파일 이름
    level=logging.DEBUG,        # 로그 수준 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # 로그 포맷
)

# 로그 메시지 기록 예시
logging.info('스크립트 실행 시작')
try:
    # 실행할 코드
    logging.info('작업 실행 중...')
    # 예: 어떤 작업 수행
    result = 10 / 0  # 예외를 발생시키는 코드 (예: ZeroDivisionError)
    logging.info('작업 완료')
except Exception as e:
    logging.error(f'오류 발생: {e}')



# ===================param 생성====================
back_date = 0 # 며칠 전에서부터 시작할 건지
today_date = -1
times = 5  # 몇 번 돌린 건지


# status = 'param' # 'update' : 오늘 자 시그널 업데이트 / 'test' : 필요한 데이터 출력해서 신규 csv로 저장
# new = False #현재 종가를 입력해서 갱신할지 : True, False

# stock_list = ['IGV', '041510.KQ']
stock_list = ['NVDY']
# stock_list = ['AAPL', 'AMZN', 'ARKG', 'DIS', 'GOOGL', 'IONQ', 'KO', 'MCD', 'MSFT', 'NVDA', 'PLTR', 'QQQ', 'QQQM', 'QUBT', 'RKLB', 'SCHD', 'SPY', 'TSM', 'UBER', 'XBI']
# stock_list = ['PLTR']

for stock in stock_list:
    update_param(stock, 'NO', back_date, today_date, times)
    # signal_with_param(stock, 'NO', back_date, today_date, times, status, False)

