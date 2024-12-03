import talib
from kibon import cal_mdd, backtesting, get_parameters
import pygad
import time

# DB
from db import get_param

# 시그널 변환 함수

def make_sma_sig(df, sma, period, sig_column):
  # 이동평균 계산
  df[sma] = talib.SMA(df['Close'], timeperiod=period)
  # 포매팅
  df[sma].astype(float)

  # 시그널 생성
  df[sig_column] = 0  # 초기 시그널은 0으로 설정

  # 이동평균 상향 돌파 시그널
  # 전일 : mktcap < sma / 후일(현재 지점) : mktcap > sma
  df.loc[(df['Close'].shift() < df[sma].shift()) & (df['Close'] > df[sma]), sig_column] = 1

  # 이동평균 하향 돌파 시그널
  # 전일 : mktcap > sma / 후일(현재 지점) : mktcap < sma
  df.loc[(df['Close'].shift() > df[sma].shift()) & (df['Close'] < df[sma]), sig_column] = -1

  return df


# ga 최적화 함수
def ga_sma_optimize(df):
  #-----------fitness 정의 (함수 수정 완료)--------------------------------------------
  def sma_fitness_func(ga_instance, solution, solution_idx):
    # mdd계산
    mdd = cal_mdd(df)

    # 솔루션 : timeperiod
    period = solution[0]

    # 생성된 매개변수로 시그널 생성
    make_sma_ga_signal(df, period)

    # 생성된 시그널을 바탕으로 수익률 계산
    profit = backtesting(df, 'sma')[0]
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
                        fitness_func=sma_fitness_func,
                        parent_selection_type=parent_selection_type,
                        mutation_type=mutation_type,
                        mutation_num_genes=mutation_num_genes,
                        # on_generation=callback_generation,
                        )

  # 유전자 알고리즘 실행
  print("SMA 유전자 알고리즘 실행 ......")
  ga_instance.run()

  best_solution = ga_instance.best_solution()  # 전체 최적 결과(현재 까지 찾은 최적 해와 적합도중 best를 출력)
  best_sma_timeperiod = best_solution[0][0]

  return best_sma_timeperiod

#-------------------------------------ga 바탕 rsi signal----------------------------------------------
# 시그널 생성 함수
def make_sma_ga_signal(df, best_sma_timeperiod):
  # 이동평균 계산
  df['ga_sma'] = talib.SMA(df['Close'], timeperiod=best_sma_timeperiod)

  # 시그널 생성
  df['ga_sma_signal'] = 0  # 초기 시그널 값을 0으로 설정


  # 이동평균 상향 돌파 시그널
  # 전일 : mktcap < ga_sma / 후일(현재 지점) : mktcap > ga_sma
  df.loc[(df['Close'].shift() < df['ga_sma'].shift()) & (df['Close'] > df['ga_sma']), 'ga_sma_signal'] = 1

  # 이동평균 하향 돌파 시그널
  # 전일 : mktcap > ga_sma / 후일(현재 지점) : mktcap < ga_sma
  df.loc[(df['Close'].shift() > df['ga_sma'].shift()) & (df['Close'] < df['ga_sma']), 'ga_sma_signal'] = -1

  return df['ga_sma_signal']


#---------------------매개변수 수렴확인---------------------
def best_sma_profit(df, stock, type):
      # best_sma_period = 36.0  # AAPL
      # best_sma_period = 23.0  # GOOGL
      # best_sma_period = 19.0    # IONQ
      # best_sma_period = 33.0    # MSFT
      # best_sma_period = 11.0   # NVDA
      # best_sma_period = 33.0  # RKLB
      # best_sma_period = 29.0  # TSM

  if type == 'GA' or type == 'ALL':
      parameters = get_param(stock, 'sma')
      # print(f"parameters : {parameters}")
      best_sma_period = parameters[2]
      # print(f"best_sma_period : {best_sma_period}")

  elif type == 'ESN' or type == 'NO':
      # 매개변수 최적화하기
      best_sma_period = ga_sma_optimize(df)

  # 수익률 확인 및 최적 매개변수로 시그널 갱신
  make_sma_ga_signal(df, best_sma_period)
  fit_returns = backtesting(df,'sma')[0]

  return best_sma_period, fit_returns, df


# 최종적으로 모듈화된 수익률 도출 함수

def sma_test(df, stock, type):
  # 기본 시그널 생성
  make_sma_sig(df, 'sma', 20, 'sma_signal')

  # GA로 필요한 매개변수 최적화
  sma_time, sma_fit, sma_df = best_sma_profit(df, stock, type)

  return sma_df['ga_sma_signal'], sma_time, sma_fit