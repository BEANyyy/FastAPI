import logging
from stock import signal_with_param
from db import upload_signals_to_db
from kibon import backtesting, new_backtesting, show_graph
import pandas as pd


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



def compare_profits(stock, model):
    df = pd.read_csv(f'STOCK/{stock}/{model}.csv')
    print(df)

    result = backtesting(df, 'result')
    # result = new_backtesting(df, 'result')
    print("result : ", result)

    # show_graph(df, 'RESULT', result[1], result[2])

    return result[0]


# ===================오늘 자 시그널 업데이트====================
# stock_list = ['AAPL', 'AMZN', 'ARKG', 'DIS', 'GOOGL', 'IONQ', 'KO', 'MCD', 'MSFT', 'NVDA', 'QQQ', 'QQQM', 'QUBT', 'RKLB', 'SCHD', 'SPY', 'TSM', 'UBER', 'XBI']
stock_list = ['AAPL']
model_list = [('ESN', 1), ('ESN', 3), ('GA', 1)]

for stock in stock_list:
    for model in model_list:
        signal_with_param(stock, model[0], 0, -1, model[1], 'update', False)

result_dict = {}

for stock in stock_list:
    max_type = None
    max_profits = 0
    for model in model_list:
        result_profits = compare_profits(stock, model[0])

        if result_profits > max_profits:
            max_type = model[0]
            max_profits = result_profits

    # stock을 키로 하고 (max_type, max_profits)를 값으로 사전에 추가
    result_dict[stock] = (max_type, max_profits)

print(result_dict)


#---------db에 저장---------
# 최고 수익률을 내는 모델만 골라서 csv 파일을 signals 테이블에 저장함.
# 한 달 동안 해당 임계치에 맞춰서 업데이트하고, Spring에는 DB에서 바로 select문으로 가져와야함. (시간 단축을 위함)

print(len(stock_list))
for stock in stock_list:
    model_name = result_dict[stock][0]
    if model_name == None:
        model_name = 'ESN_3'

    # Example usage
    csv_path = f'STOCK/{stock}/{model_name}.csv'  # Replace with your CSV file path

    upload_signals_to_db(csv_path)