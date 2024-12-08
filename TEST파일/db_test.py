import pandas as pd
from stock import signal_with_param
from db import fetch_data, connect_to_db
import pymysql
# ===================Spring에 넘길 모델====================
def realtime_predict(stock):
    # if __name__ == "db":
    if __name__ == "__main__":
        # 연결 정보 설정
        host = "localhost"  # MySQL 서버 호스트 (로컬의 경우 localhost)
        user = "root"  # MySQL 사용자 이름
        password = "1234"  # MySQL 비밀번호
        database = "stock_db"  # 데이터베이스 이름

        # 데이터베이스 연결
        connection = connect_to_db(host, user, password, database)

        if connection:
            # 조건 값을 사용하여 데이터 가져오기
            table_name = "model_result"  # 조회할 테이블 이름
            column_name = "stock_name"  # 조건을 적용할 컬럼 이름
            condition_value = stock  # 조건 값

            data = fetch_data(connection, table_name, column_name, condition_value)

            # 모델 이름 추출하기
            if data:
                model_name = data[0][2]

        print("--------------------------------")

    print(f"stock : {stock}, model_name : {model_name}")
    times = 1
    if model_name == 'ESN_3':
        model_name = 'ESN'
        times = 3

    result = signal_with_param(stock, model_name, 0, -1, times, 'no', True)

    return result



# ===================Spring에 넘길 모델====================
def today_predict(stock):
    # if __name__ == "db":
    if __name__ == "__main__":
        # 연결 정보 설정
        host = "localhost"  # MySQL 서버 호스트 (로컬의 경우 localhost)
        user = "root"  # MySQL 사용자 이름
        password = "1234"  # MySQL 비밀번호
        database = "stock_db"  # 데이터베이스 이름

        # 데이터베이스 연결
        connection = connect_to_db(host, user, password, database)

        if connection:
            # 조건 값을 사용하여 데이터 가져오기
            table_name = "signals"  # 조회할 테이블 이름
            column_name = "stock_name"  # 조건을 적용할 컬럼 이름
            condition_value = stock  # 조건 값

            data = fetch_data(connection, table_name, column_name, condition_value)

            # 가져온 데이터 출력
            if data:
                # 날짜 기준으로 정렬
                sorted_data = sorted(data, key=lambda x: x[-1],
                                     reverse=True)  # reverse=True : 가장 최신 값 / reverse=True : 가장 옛날 값 출력됨

                # 가장 최신 데이터
                latest_data = sorted_data[0]

                print("------------------------")
                print(f"현재 날짜")
                print(latest_data[-4])
                signal = latest_data[-2]
                print("------------------------")


        print("--------------------------------")




    return signal


# stock_list = ['AAPL', 'ARKG', 'AMZN', 'DIS', 'GOOGL',
#               'IONQ', 'KO', 'MCD', 'MSFT', 'NVDA',
#               'QQQ', 'QQQM', 'QUBT', 'RKLB', 'SCHD',
#               'SPY', 'TSM', 'UBER','XBI']


stock_list = ['AAPL', 'AMZN','GOOGL']

for stock in stock_list:
    result = today_predict(stock)