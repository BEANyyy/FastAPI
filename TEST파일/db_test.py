from db import connect_to_db, fetch_data, fetch_latest_data, get_param



# host = "localhost"  # MySQL 서버 호스트 (로컬의 경우 localhost)
# user = "root"  # MySQL 사용자 이름
# password = "1234"  # MySQL 비밀번호
# database = "stock_db"  # 데이터베이스 이름
#
# # 데이터베이스 연결
# connection = connect_to_db(host, user, password, database)
#
# # table_name = "rsi"
# # column_name = "stock_id"
# # condition_value = "IGV"
# #
# # data = fetch_data(connection, table_name, column_name, condition_value)
# #
# #
# # # 가져온 데이터 출력
# # if data:
# #     # 날짜 기준으로 정렬
# #     sorted_data = sorted(data, key=lambda x: x[-1], reverse=True)  # reverse=True : 가장 최신 값 / reverse=True : 가장 옛날 값 출력됨
# #
# #     # 가장 최신 데이터
# #     latest_data = sorted_data[0]
# #
# #     print("------------------------")
# #     print(f"현재 {table_name} param의 생성 기준 날짜")
# #     print(latest_data[-1])
# #     print("------------------------")
# #
# #     result_param = latest_data
# #     print(result_param)
#
# get_param('IGV', 'rsi')



import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
from dateutil.relativedelta import relativedelta

# # 미국 뉴욕 시간대 설정
# us_time_zone = pytz.timezone('America/New_York')
# stock = "AAPL"
# # print(f"현재 최적화하는 주식 :  {stock}")
# # # 현재 시간(미국 뉴욕 시간대 기준) 가져오기
# # today = datetime.now(us_time_zone).date()
# # # end = today + relativedelta(days=1)
# # end = today - relativedelta(days=1)
# # print("미국 뉴욕 시간 기준 현재 날짜:", end)
# #
# # # 6개월 전 날짜 계산
# # start = end - relativedelta(months=12)
# #
# # print("미국 뉴욕 시간 기준 1년 전 날짜:", start)
# df = yf.download(stock)
# df = df.reset_index()
# df.columns = df.columns.droplevel(1)
#
# print(df)

import yfinance as yf

# 간단한 예시 코드
aapl = yf.download('AAPL', start='2023-01-01', end='2023-12-31', interval="1d")
print(aapl)
