# model을 정하기 : db에서 빼오기
from db import get_param
from kibon import save_graph, backtesting
from stock import realtime_signal
import pandas as pd
import matplotlib.pyplot as plt

stock = 'AAPL'
model = get_param(stock, 'model_result')[2]

# 추가할 데이터
print("😋😋😋실시간 분석 시작😋😋😋")
print("현재 시점에서의 종가를 입력하세요")
# close = float(input())
close = 225


times = 1

if model == 'ESN_3':
    model = 'ESN'
    times = 3

result, new_row = realtime_signal(stock,  model, 0, -1, times, close)
df = pd.read_csv(f'STOCK/{stock}/{model}.csv')


# 여기에 result_df를 더해야 할 것 같숭.


# 'Date' 열을 Python의 datetime 객체로 변환
new_row['Date'] = pd.to_datetime(new_row['Date']).astype(str)

# 기존 DataFrame과 새로운 행을 concat으로 합치기
df = pd.concat([df, new_row], ignore_index=True)
print(df)

graph = backtesting(df, 'result')
# result = new_backtesting(df, 'result')
print("profits : ", graph)

fig = save_graph(df, 'RESULT', graph[1], graph[2], graph[0])

# Figure 활성화 및 표시
plt.figure(fig.number)  # 생성된 Figure 활성화
plt.show()