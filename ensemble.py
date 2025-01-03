import pandas as pd
from kibon import backtesting, show_graph, split_data

file_path = 'STOCK/AAPL/'

df1 = pd.read_csv(file_path + 'ALL.csv')
df2 = pd.read_csv(file_path + 'ESN.csv')
df3 = pd.read_csv(file_path + 'ESN.csv')
df4 = pd.read_csv(file_path + 'GA.csv')

# 데이터프레임 리스트
dfs = [df1, df2, df3, df4]

# # 모든 데이터프레임 합치기
# combined_df = pd.concat(dfs)
#
# # 'date'별로 -1, 0, 1의 개수를 배열로 계산
# value_counts_by_date = (
#     combined_df.groupby('Date')['pred_sig']
#     .apply(lambda x: [x.tolist().count(-1), x.tolist().count(0), x.tolist().count(1)])
#     .reset_index()
#     .rename(columns={'pred_sig': 'counts'})
# )
#
# # 'date'별로 가장 많이 나온 'pred_sig' 계산
# def resolve_most_common_pred_sig(counts):
#     max_count = max(counts)
#     # 가장 많이 나온 값이 여러 개일 경우 0으로 설정
#     if counts.count(max_count) > 1:
#         return 0
#     # 그렇지 않으면 가장 큰 빈도를 가진 값의 인덱스(-1, 0, 1 중 하나 반환)
#     return [-1, 0, 1][counts.index(max_count)]
#
# most_common_pred_sig_by_date = (
#     value_counts_by_date
#     .assign(most_common_pred_sig=value_counts_by_date['counts'].apply(resolve_most_common_pred_sig))
# )
#
# # 'result' 데이터프레임 생성: combined_df에 most_common_pred_sig와 counts 추가
# result_df = combined_df.merge(
#     most_common_pred_sig_by_date[['Date', 'most_common_pred_sig', 'counts']],
#     on='Date',
#     how='left'
# )
#
# # result에서 df1의 길이만 남기기
# df1_length = len(df1)
# result_df = result_df.iloc[:df1_length].reset_index(drop=True)
#
#
# # 결과 확인
# print(result_df.columns)
# print(len(result_df))
# # result_df.to_csv(file_path+'ensemble.csv')
# print(result_df[['Date', 'pred_sig', 'most_common_pred_sig', 'counts']].tail(10))
# # print(df1)
# # print(df2)
# # print(df3)
# # print(df4)
#
# # print(f"len df1 : {len(df1)}")
# # print(f"len df2 : {len(df2)}")
# # print(f"len df3 : {len(df3)}")
# # print(f"len df4 : {len(df4)}")
# # print(f"len result_df : {len(result_df)}")
#
#
#
# # 'pred_sig' 열 삭제
# result_df = result_df.drop(columns=['pred_sig'])
#
# # 'most_common_pred_sig' 열을 'pred_sig'로 이름 변경
# result_df = result_df.rename(columns={'most_common_pred_sig': 'pred_sig'})


result_df = pd.read_csv(file_path + 'ensemble.csv')
print(result_df)
result = backtesting(result_df, 'result')
print("result : ", result[0])
show_graph(result_df, 'RESULT', result[1], result[2])