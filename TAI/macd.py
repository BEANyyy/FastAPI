import talib
from kibon import cal_mdd, backtesting, get_parameters
import pygad
import time

# DB
from db import get_param

# 시그널 변환 함수
# 일반적인 기준으로 MACD 시그널 계산하는 함수 (fast_period=12, slow_period=26, signal_period=9)

#시그널 생성함수 모듈화
def make_macd_sig(df, macd, signal_line, fast_period, slow_period, sig_period, sig_column):
  macdval =talib.MACD(df['Close'],fast_period, slow_period, sig_period)

  df[macd] = macdval[0]  # macd 값
  df[signal_line] = macdval[1]  # macd 시그널 라인값

  # 시그널 생성
  df[sig_column] = 0  # 초기 시그널은 0으로 설정

  # 골든크로스
  # 전일 : macd < signal line / 후일(현재 지점) : macd > signal line (macd가 치고 올라가는-매수 타이밍)
  df.loc[(df[macd].shift() < df[signal_line].shift()) & (df[macd] > df[signal_line]), sig_column] = 1
  # 데드크로스
  # 전일 : macd > signal line / 후일(현재 지점) : macd < signal line (macd가 아래로 내려가는-매도 타이밍)
  df.loc[(df[macd].shift() > df[signal_line].shift()) & (df[macd] < df[signal_line]), sig_column] = -1

  return df



# ga 최적화 함수
def ga_macd_optimize(df):
  #-----------fitness 정의 (함수 수정 완료)--------------------------------------------
  def macd_fitness_func(ga_instance, solution, solution_idx):
      #mdd계산
      mdd = cal_mdd(df)

      # 솔루션 : 단기, 장기, 시그널 기간
      fast_period = solution[0]
      slow_period = solution[1]
      signal_period = solution[2]

      # 생성된 매개변수로 시그널 생성
      make_macd_ga_signal(df, fast_period, slow_period, signal_period)

      # 생성된 시그널을 바탕으로 수익률 계산
      profit = backtesting(df, 'macd')[0]

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
                        num_genes=3,  # 매개변수 수
                       gene_space = [range(3, 20), range(15, 30), range(3, 15)], # 매개변수 범위제한 -> 해당 범위 잘 적용됨을 확인 함
                        fitness_func=macd_fitness_func,
                        parent_selection_type=parent_selection_type,
                        mutation_type=mutation_type,
                        mutation_num_genes = mutation_num_genes,
                        # on_generation=callback_generation,
                        )

  # 유전자 알고리즘 실행
  print("MACD 유전자 알고리즘 실행 ......")
  ga_instance.run()


  best_solution = ga_instance.best_solution() # 전체 최적 결과(현재 까지 찾은 최적 해와 적합도중 best를 출력)
  best_fastperiod = best_solution[0][0]
  best_slowperiod = best_solution[0][1]
  best_signalperiod = best_solution[0][2]

  return best_fastperiod, best_slowperiod, best_signalperiod


#-------------------------------------ga 바탕 rsi signal----------------------------------------------
# 시그널 생성 함수
def make_macd_ga_signal(df,  best_fastperiod, best_slowperiod, best_signalperiod):
  #macd값 계산
  macdval, macdsignal, macdhist = talib.MACD(df['Close'], best_fastperiod, best_slowperiod, best_signalperiod)
  df['ga_macd'] = macdval
  df['ga_signal_line'] = macdsignal

  # 시그널 생성
  df['ga_macd_signal'] = 0  # 초기 시그널 값을 0으로 설정

  # macd 곡선이 시그널 곡선 상향 골든크로스 => 매수시점(1)
  # 전일 : macd < signal line / 후일(현재 지점) : macd > signal line (macd가 치고 올라가는-매수 타이밍)
  df.loc[(df['ga_macd'].shift() < df['ga_signal_line'].shift()) & (df['ga_macd'] > df['ga_signal_line']), 'ga_macd_signal'] = 1

  # macd 곡선이 시그널 곡선 하향 데드크로스 => 매도시점(-1)
  # 전일 : macd > signal line / 후일(현재 지점) : macd < signal line (macd가 아래로 내려가는-매도 타이밍)
  df.loc[(df['ga_macd'].shift() > df['ga_signal_line'].shift()) & (df['ga_macd'] < df['ga_signal_line']), 'ga_macd_signal'] = -1

  return df['ga_macd_signal']


#---------------------매개변수 수렴확인---------------------

def best_macd_profit(df, stock, type):
        # best_fastperiod, best_slowperiod, best_signalperiod = 17.0, 28.0, 13.0  # AAPL
        # best_fastperiod, best_slowperiod, best_signalperiod = 13.0, 19.0, 14.0  # GOOGL
        # best_fastperiod, best_slowperiod, best_signalperiod = 3.0, 15.0, 4.0    # IONQ
        # best_fastperiod, best_slowperiod, best_signalperiod = 3.0, 26.0, 3.0    # MSFT
        # best_fastperiod, best_slowperiod, best_signalperiod = 11.0, 23.0, 7.0 # NVDA
        # best_fastperiod, best_slowperiod, best_signalperiod = 18.0, 29.0, 13.0  # RKLB
        # best_fastperiod, best_slowperiod, best_signalperiod = 4.0, 25.0, 6.0  # TSM

    if type == 'GA' or type == 'ALL':
        # 최적화된 매개변수 출력 : ESN 최적화한 상태에서 10번 돌렸을 때 가장 높은 profit이 나온 임계치를 저장함.
        parameters = get_param(stock, 'macd')
        # print(f"parameters : {parameters}")
        best_fastperiod, best_slowperiod, best_signalperiod = parameters[2], parameters[3], parameters[4]
        # print(f"best_fastperiod, best_slowperiod, best_signalperiod : {best_fastperiod}, {best_slowperiod}, {best_signalperiod}")

    elif type == 'ESN'  or type == 'NO':
        # 매개변수 최적화하기
        best_fastperiod, best_slowperiod, best_signalperiod = ga_macd_optimize(df)


    # 수익률 확인 및 최적 매개변수로 시그널 갱신
    make_macd_ga_signal(df, best_fastperiod, best_slowperiod, best_signalperiod)
    fit_returns = backtesting(df, 'macd')[0]

    # 최적화된 매개변수 출력, 최적화된 수익 출력(mdd로 나눈 값 아님 주의 단순 수익률)
    return best_fastperiod, best_slowperiod, best_signalperiod, fit_returns, df



# 최종적으로 모듈화된 수익률 도출 함수
def merged_macd_test(df, stock, type):
  #train 데이터에 기본 시그널 생성
  make_macd_sig(df, 'macd', 'signal_line', 12, 26, 9, 'macd_signal')

  #ga를 이용하여 매개변수 최적화하기
  macd_fastp, macd_slowp, macd_sigp, macd_fit, macd_df = best_macd_profit(df, stock, type)

  return macd_df['ga_macd_signal'], macd_fastp, macd_slowp, macd_sigp, macd_fit