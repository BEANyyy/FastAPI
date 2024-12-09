# model을 정하기 : db에서 빼오기
from db import get_param
from kibon import save_graph, backtesting
from stock import realtime_signal
import pandas as pd
import matplotlib.pyplot as plt

stock = 'GOOGL'
model = get_param(stock, 'model_result')[2]

# 추가할 데이터
print("😋😋😋실시간 분석 시작😋😋😋")
print("현재 시점에서의 종가를 입력하세요")
close = float(input())


times = 1

if model == 'ESN_3':
    model = 'ESN'
    times = 3

result = realtime_signal(stock,  model, 0, -1, times, close)
df = pd.read_csv(f'STOCK/{stock}/{model}.csv')
print(df)

graph = backtesting(df, 'result')
# result = new_backtesting(df, 'result')
print("profits : ", graph)

fig = save_graph(df, 'RESULT', graph[1], graph[2], graph[0])

# Figure 활성화 및 표시
plt.figure(fig.number)  # 생성된 Figure 활성화
plt.show()