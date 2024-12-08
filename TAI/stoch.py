import talib
from kibon import cal_mdd, backtesting
import pygad
import time

# DB
from db import get_param

# # 시그널 변환 함수
# def add_ratio(df):
#   df['high_ratio'] = (df['MKTCAP'] * df['High']) / df['Close'] #high값을 MKTCAP에 상응하는 비율로 맞춘 열 추가
#   df['low_ratio'] = (df['MKTCAP'] * df['Low']) / df['Close']  #low값을 MKTCAP에 상응하는 비율로 맞춘 열 추가
#   return df

#시그널 생성함수 모듈화
def make_stoch_sig(df, k_line, d_line, fast_period, slow_period, sig_column):
  stoch = talib.STOCH(df["High"], df["Low"], df["Close"], fastk_period=fast_period, slowk_period=slow_period, slowd_period=slow_period)

  df[k_line] = stoch[0]  # K선 값
  df[d_line] = stoch[1]  # D선 값

  # 시그널 생성
  df[sig_column] = 0  # 초기 시그널은 0으로 설정

  # K선이 D선 위로 돌파시 골든크로스 => 매수시점(1)
  # 전일: K_line < D_line / 후일(현재 지점) :  K_line > D_line
  df.loc[(df[k_line].shift() <= df[d_line].shift()) & (df[k_line] > df[d_line]), sig_column] = 1

  # K선이 D선 아래로 돌파시 데드크로스 => 매도시점(-1)
  # 전일: K_line > D_line / 후일(현재 지점) :  K_line < D_line
  df.loc[(df[k_line].shift() >= df[d_line].shift()) & (df[k_line] < df[d_line]), sig_column] = -1

  return df




# ga 최적화 함수
def ga_stoch_optimize(df):
  #-----------fitness 정의 (함수 수정 완료)--------------------------------------------
  def stoch_fitness_func(ga_instance, solution, solution_idx):
      #mdd계산
      mdd = cal_mdd(df)

      # 솔루션 : K선 결정 범위 기간 period_K, K선을 이동평균하는 기간 period_D
      period_K = solution[0]
      period_D = solution[1]

      # 생성된 매개변수로 시그널 생성
      make_stoch_ga_signal(df, period_K, period_D)

      # 생성된 시그널을 바탕으로 수익률 계산
      profit = backtesting(df, 'stoch')[0]

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
                        num_genes=2,  # 매개변수 수
                        gene_space = [range(5, 28), range(3, 10)], # 매개변수 범위제한 -> 해당 범위 잘 적용됨을 확인 함
                        fitness_func=stoch_fitness_func,
                        parent_selection_type=parent_selection_type,
                        mutation_type=mutation_type,
                        mutation_num_genes=mutation_num_genes,
                        # on_generation=callback_generation,
                        )

  # 유전자 알고리즘 실행
  print("STOCH 유전자 알고리즘 실행 ......")
  ga_instance.run()

  best_solution = ga_instance.best_solution() # 전체 최적 결과(현재 까지 찾은 최적 해와 적합도중 best를 출력)
  best_period_K = best_solution[0][0]
  best_period_D = best_solution[0][1]

  return best_period_K, best_period_D

#best_period_K, best_period_D = ga_optimize()
#print(f"Best Parameters: K = {best_period_K}, D = {best_period_D}")

#-------------------------------------ga 바탕 rsi signal----------------------------------------------
# 시그널 생성 함수
def make_stoch_ga_signal(df, best_period_K, best_period_D):
  #stoch값 계산
  stoch = talib.STOCH(df["High"], df["Low"], df["Close"], fastk_period=best_period_K, slowk_period=best_period_D, slowd_period=best_period_D)
  df['ga_k_line'] = stoch[0]  # K선 값
  df['ga_D_line'] = stoch[1]  # D선 값

  # 시그널 생성
  df['ga_stoch_signal'] = 0  # 초기 시그널 값을 0으로 설정

  # K선이 D선 위로 돌파시 골든크로스 => 매수시점(1)
  # 전일: K_line < D_line / 후일(현재 지점) :  K_line > D_line
  df.loc[(df['ga_k_line'].shift() <= df['ga_D_line'].shift()) & (df['ga_k_line'] > df['ga_D_line']), 'ga_stoch_signal'] = 1

  # K선이 D선 아래로 돌파시 데드크로스 => 매도시점(-1)
  # 전일: K_line > D_line / 후일(현재 지점) :  K_line < D_line
  df.loc[(df['ga_k_line'].shift() >= df['ga_D_line'].shift()) & (df['ga_k_line'] < df['ga_D_line']), 'ga_stoch_signal'] = -1

  return df['ga_stoch_signal']

#---------------------매개변수 수렴확인---------------------

def best_stoch_profit(df, stock, type):
      # best_period_K, best_period_D = 26.0, 8.0    # AAPL
      # best_period_K, best_period_D = 14.0, 9.0  # GOOGL
      # best_period_K, best_period_D = 5.0, 7.0  # IONQ
      # best_period_K, best_period_D = 10.0, 4.0  # MSFT
      # best_period_K, best_period_D = 6.0, 3.0   # NVDA
      # best_period_K, best_period_D = 23.0, 3.0    # RKLB
      # best_period_K, best_period_D = 6.0, 7.0    # TSM

  if type == 'GA' or type == 'ALL':
      # 최적화된 매개변수 출력 : ESN 최적화한 상태에서 10번 돌렸을 때 가장 높은 profit이 나온 임계치를 저장함.
      parameters = get_param(stock, 'stoch')
      # print(f"parameters : {parameters}")
      best_period_K, best_period_D = parameters[2], parameters[3]
      # print(f"best_period_K, best_period_D : {best_period_K}, {best_period_D}")

  elif type == 'ESN' or type == 'NO':
      # 매개변수 최적화하기
      best_period_K, best_period_D = ga_stoch_optimize(df)


  # 수익률 확인 및 최적 매개변수로 시그널 갱신
  make_stoch_ga_signal(df, best_period_K, best_period_D)
  fit_returns = backtesting(df, 'stoch')[0]

  # 최적화된 매개변수 출력, 최적화된 수익 출력(mdd로 나눈 값 아님 주의 단순 수익률)
  return best_period_K, best_period_D, fit_returns, df



# 최종적으로 모듈화된 수익률 도출 함수
def stoch_test(df, stock, type):
  #high_ratio, low_ratio 열추가(스토캐스틱만)
  #df = add_ratio(df)

  # 기본 시그널 생성
  make_stoch_sig(df, 'k_line', 'd_line', 14, 3, 'stoch_signal')

  # ga를 이용하여 매개변수 최적화하기
  stoch_k, stoch_d, stoch_fit, stoch_df = best_stoch_profit(df, stock, type)

  return stoch_df['ga_stoch_signal'], stoch_k, stoch_d, stoch_fit