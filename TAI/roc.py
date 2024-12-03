import talib
from kibon import cal_mdd, backtesting, get_parameters
import pygad
import time

# DB
from db import get_param

# 시그널 변환 함수

def make_roc_sig(df, roc, period, sig_column):
  # ROC 계산 : 보통 10으로 설정
  df[roc] = talib.ROC(df['Close'], timeperiod=period)

  # 시그널 생성
  df[sig_column] = 0  # 초기 시그널은 0으로 설정

  # 이동평균 상향 돌파 시그널
  # 전일 : mktcap < 0 / 후일(현재 지점) : mktcap > 0
  df.loc[(df[roc].shift() < 0) & (df[roc] > 0), sig_column] = 1

  # 이동평균 하향 돌파 시그널
  # 전일 : mktcap > 0 / 후일(현재 지점) : mktcap < 0
  df.loc[(df[roc].shift() > 0) & (df[roc] < 0), sig_column] = -1

  return df


# ga 최적화 함수
def ga_roc_optimize(df):
  #-----------fitness 정의 (함수 수정 완료)--------------------------------------------
  def roc_fitness_func(ga_instance, solution, solution_idx):
      # mdd 계산
      mdd = cal_mdd(df)

      # 솔루션 : timeperiod
      period = solution[0]

      # 생성된 매개변수로 시그널 생성
      make_roc_ga_signal(df, period)

      # 생성된 시그널을 바탕으로 수익률 계산
      profit = backtesting(df, 'roc')[0]
      fitness = 0.9*profit + 0.1*mdd  # 유전자 적합도 판단 기준(=다음 세대를 위한 우월 개체 선정에 쓰일 가산점)

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
  num_generations = 10  # 세대 수
  num_parents_mating = 10  # 부모 개체 수
  sol_per_pop = 20  # 개체 수
  parent_selection_type = "rws"
  mutation_type = "random"
  mutation_num_genes = 1

  # PyGAD GA 객체 생성
  ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        sol_per_pop=sol_per_pop,
                        num_genes=1,  # 매개변수 수
                        gene_space = range(3, 60),
                        fitness_func=roc_fitness_func,
                        parent_selection_type=parent_selection_type,
                        mutation_type=mutation_type,
                        mutation_num_genes=mutation_num_genes,
                        # on_generation=callback_generation,
                        )

  # 유전자 알고리즘 실행
  print("ROC 유전자 알고리즘 실행 ......")
  ga_instance.run()

  best_solution = ga_instance.best_solution()  # 전체 최적 결과(현재 까지 찾은 최적 해와 적합도중 best를 출력)
  best_roc_timeperiod = best_solution[0][0]

  return best_roc_timeperiod

#-------------------------------------ga 바탕 roc signal----------------------------------------------
# 시그널 생성
def make_roc_ga_signal(df, best_roc_timeperiod):
  # 이동평균 계산
  df['ga_roc'] = talib.ROC(df['Close'], timeperiod=best_roc_timeperiod)

  # 시그널 생성
  df['ga_roc_signal'] = 0  # 초기 시그널 값을 0으로 설정

  # 이동평균 상향 돌파 시그널
  # 전일 : mktcap < 0 / 후일(현재 지점) : mktcap > 0
  df.loc[(df['ga_roc'].shift() < 0) & (df['ga_roc'] > 0), 'ga_roc_signal'] = 1

  # 이동평균 하향 돌파 시그널
  # 전일 : mktcap > 0 / 후일(현재 지점) : mktcap < 0
  df.loc[(df['ga_roc'].shift() > 0) & (df['ga_roc'] < 0), 'ga_roc_signal'] = -1

  return df['ga_roc_signal']

#---------------------매개변수 수렴확인---------------------
def best_roc_profit(df, stock, type):
      # best_roc_period = 40.0  # AAPL
      # best_roc_period = 59.0  # GOOGL
      # best_roc_period = 3.0    # IONQ
      # best_roc_period = 36.0    # MSFT
      # best_roc_period = 21.0    # NVDA
      # best_roc_period = 36.0  # RKLB
      # best_roc_period = 16.0  # TSM

  if type == 'GA' or type == 'ALL':
      # 최적화된 매개변수 출력 : ESN 최적화한 상태에서 10번 돌렸을 때 가장 높은 profit이 나온 임계치를 저장함.
      parameters = get_param(stock, 'roc')
      # print(f"parameters : {parameters}")
      best_roc_period = parameters[2]
      # print(f"best_roc_period : {best_roc_period}")

  elif type == 'ESN'  or type == 'NO':
      # 매개변수 최적화하기
      best_roc_period = ga_roc_optimize(df)



  # 수익률 확인 및 최적 매개변수로 시그널 갱신
  make_roc_ga_signal(df, best_roc_period)
  fit_returns = backtesting(df, 'roc')[0]

  return best_roc_period, fit_returns, df



# 최종적으로 모듈화된 수익률 도출 함수
def roc_test(df, stock, type):
  # train 데이터에 기본 시그널 생성
  make_roc_sig(df, 'roc', 10, 'roc_signal')

  # GA로 필요한 매개변수 최적화
  roc_time, roc_fit, roc_df = best_roc_profit(df, stock, type)

  return roc_df['ga_roc_signal'], roc_time, roc_fit