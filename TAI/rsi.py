import talib
from kibon import cal_mdd, backtesting
import pygad
import time

# DB
from db import get_param




# 시그널 변환 함수

def make_rsi_sig(df, rsi, thres_sell, thres_buy, period, sig_column):
  # RSI 계산
  #df[rsi] = ta.momentum.rsi(df['MKTCAP'])
  df[rsi] = talib.RSI(df['Close'], timeperiod=period) # RSI 기간 설정 (일반적으로 14일을 사용)

  # 시그널 생성
  df[sig_column] = 0  # 초기 시그널은 0으로 설정

  # RSI가 70 이상이면 Sell 시그널 (-1)
  df.loc[df[rsi] > thres_sell, sig_column] = -1

  # RSI가 30 이하이면 Buy 시그널 (1)
  df.loc[df[rsi] < thres_buy, sig_column] = 1

  return df

# ga 최적화 함수
def ga_rsi_optimize(df):
  #-----------fitness 정의 (함수 수정 완료)--------------------------------------------
  def rsi_fitness_func(ga_instance, solution, solution_idx):
      #mdd계산
      mdd = cal_mdd(df)

      # 솔루션 : 매도 임계값, 매수 임계값, 고려 기간
      rsi_sell_threshold = solution[0]
      rsi_buy_threshold = solution[1]
      rsi_timeperiod = solution[2]

      #생성된 매개변수로 시그널 생성
      make_rsi_ga_signal(df, rsi_sell_threshold, rsi_buy_threshold, rsi_timeperiod)

      #생성된 시그널을 바탕으로 수익률 계산
      profit = backtesting(df, 'rsi')[0]
      fitness = 0.9*profit + 0.1*mdd  #유전자 적합도 판단 기준(=다음 세대를 위한 우월 개체 선정에 쓰일 가산점)

      return fitness

  def callback_generation(ga_instance):
      # 현재 세대의 번호
      generation = ga_instance.generations_completed
      # 최적 해와 적합도
      best_solution, best_solution_fitness, _ = ga_instance.best_solution()

      print(f"Generation: {generation}")
      print(f"Best Solution: {best_solution}")
      print(f"Best Fitness: {round(best_solution_fitness, 2)}\n")

      # 딜레이 추가 (예: 1초)
      time.sleep(1)

  # PyGAD 옵션 설정
  num_generations = 5  # 세대 수
  num_parents_mating = 10  # 부모 개체 수
  sol_per_pop = 20  # 개체 수
  parent_selection_type = "rws"
  mutation_type = "random"
  mutation_num_genes = 1

  # PyGAD GA 객체 생성
  ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        sol_per_pop=sol_per_pop,
                        num_genes=3,  # 매개변수 수
                        # solution = [sell thres, buy thres, period]
                        gene_space = [{'low': 10, 'high': 90}, {'low': 10, 'high': 90}, {'low': 9, 'high': 28}],
                        fitness_func=rsi_fitness_func,
                        parent_selection_type=parent_selection_type,
                        mutation_type=mutation_type,
                        mutation_num_genes=mutation_num_genes,
                        # on_generation=callback_generation,
                        )

  # 유전자 알고리즘 실행
  print("RSI 유전자 알고리즘 실행 ......")
  ga_instance.run()

  best_solution = ga_instance.best_solution() # 전체 최적 결과(현재 까지 찾은 최적 해와 적합도중 best를 출력)
  best_rsi_sell_threshold = best_solution[0][0]
  best_rsi_buy_threshold = best_solution[0][1]
  best_rsi_timeperiod = best_solution[0][2]

  return best_rsi_sell_threshold, best_rsi_buy_threshold, best_rsi_timeperiod

#-------------------------------------ga 바탕 rsi signal----------------------------------------------
# 시그널 생성 함수
def make_rsi_ga_signal(df, best_rsi_sell_threshold, best_rsi_buy_threshold, best_rsi_timeperiod):

  # RSI 계산
  df['ga_rsi'] = talib.RSI(df['Close'], timeperiod=best_rsi_timeperiod)

  # 시그널 생성
  df['ga_rsi_signal'] = 0  # 초기 시그널 값을 0으로 설정

  # RSI 값이 최적화된 매도 임계값을 초과하면 시그널 값 : -1
  # df['ga_rsi'] > best_rsi_sell_threshold 조건을 충족하는 행들에 대해, ga_signal 컬럼 값을 -1로 변경
  df.loc[df['ga_rsi'] > best_rsi_sell_threshold, 'ga_rsi_signal'] = -1

  # RSI 값이 최적화된 매수 임계값을 하회하면 시그널 값 : 1
  # df['rsi'] > best_rsi_buy_threshold 조건을 충족하는 행들에 대해, ga_signal 컬럼 값을 1로 변경
  df.loc[df['ga_rsi'] < best_rsi_buy_threshold, 'ga_rsi_signal'] = 1

  return df['ga_rsi_signal']


#---------------------매개변수 수렴확인---------------------
def rsi_best_profit(df, stock, type):
  best_rsi_sell_threshold, best_rsi_buy_threshold, best_rsi_timeperiod = 0, 0, 0

  if type == 'GA' or type == 'ALL':
    # 최적화된 매개변수 출력 : ESN 최적화한 상태에서 10번 돌렸을 때 가장 높은 profit이 나온 임계치를 저장함.
    parameters = get_param(stock, 'rsi')
    # print(f"parameters : {parameters}")
    best_rsi_sell_threshold, best_rsi_buy_threshold, best_rsi_timeperiod = parameters[2], parameters[3], parameters[4]
    # print(f"best_rsi_sell_threshold, best_rsi_buy_threshold, best_rsi_timeperiod : {best_rsi_sell_threshold}, {best_rsi_buy_threshold}, {best_rsi_timeperiod}")

  elif type == 'ESN' or type == 'NO':
      # 매개변수 최적화하기
      best_rsi_sell_threshold, best_rsi_buy_threshold, best_rsi_timeperiod = ga_rsi_optimize(df)




  # 수익률 확인 및 최적 매개변수로 시그널 갱신
  make_rsi_ga_signal(df, best_rsi_sell_threshold, best_rsi_buy_threshold, best_rsi_timeperiod)
  fit_returns = backtesting(df, 'rsi')[0]

  return best_rsi_sell_threshold, best_rsi_buy_threshold, best_rsi_timeperiod, fit_returns, df



# 최종적으로 모듈화된 수익률 도출 함수
def rsi_test(df, stock, type):
  # 기본 시그널 생성
  make_rsi_sig(df, 'rsi', 70, 30, 14, 'rsi_signal')


  # GA로 필요한 매개변수 최적화
  rsi_sell, rsi_buy, rsi_time, rsi_fit, rsi_df = rsi_best_profit(df, stock, type)

  return rsi_df['ga_rsi_signal'], rsi_sell, rsi_buy, rsi_time, rsi_fit