import talib
from kibon import cal_mdd, backtesting
import pygad
import time

# DB
from db import get_param

# 시그널 변환 함수

def make_gdc_sig(df, short_ma, long_ma, short_period, long_period, sig_column):
  # 단기 이동 평균선 계산
  df[short_ma] = talib.SMA(df['Close'], timeperiod=short_period)

  # 중장기 이동 평균선 계산
  df[long_ma] = talib.SMA(df['Close'], timeperiod=long_period)

  # 포매팅
  df['short_ma'].astype(float)
  df['long_ma'].astype(float)

  # 시그널 생성
  df[sig_column] = 0 # 초기 시그널은 0으로 설정

  # 골든크로스
  # 전일 : 단기이동평균선 < 중장기 이동평균선 / 후일(현재 지점) : 단기이동평균선 > 중장기 이동평균선
  df.loc[(df[short_ma].shift() < df[long_ma].shift()) & (df[short_ma] > df[long_ma]), sig_column] = 1

  # 데드크로스
  # 전일 : 단기이동평균선 > 중장기 이동평균선 / 후일(현재 지점) : 단기이동평균선 < 중장기 이동평균선
  df.loc[(df[short_ma].shift() > df[long_ma].shift()) & (df[short_ma] < df[long_ma]), sig_column] = -1

  return df



# ga 최적화 함수
def ga_gdc_optimize(df):
  def gdc_fitness_func(ga_instance, solution, solution_idx):
      #mdd계산
      mdd = cal_mdd(df)

      # 솔루션 : 단기 이동평균 기간 L, 장기 이동평균 기간 S
      short_period = solution[0]
      long_period = solution[1]

      # 생성된 L,S 값을 이용하여 시그널 생성
      make_gdc_ga_signal(df,short_period,long_period)

      # 백테스팅으로 반환된 수익률 자체를 fitness함수의 최적화 평가 지표로 이용
      # 생성된 시그널을 바탕으로 수익률 계산
      profit = backtesting(df, 'gdc')[0]
      fitness = 0.9 * profit + 0.1 * mdd  # 유전자 적합도 판단 기준(=다음 세대를 위한 우월 개체 선정에 쓰일 가산점)
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
  num_generations = 10         # 최적화를 반복할 세대의 수
  num_parents_mating = 10       # 각 세대에서 선택되는 부모 개체의 수
  sol_per_pop = 20             # 각 세대에서 생성되는 개체의 수
  parent_selection_type = "rws" # 룰렛 휠 선택 방식
  mutation_type = "random"      # 무작위 돌연변이
  mutation_num_genes = 1

  ga_instance = pygad.GA(num_generations=num_generations,
                          num_parents_mating=num_parents_mating,
                          sol_per_pop=sol_per_pop,
                          num_genes=2,
                          gene_space=[{'low': 3, 'high': 14}, {'low': 30, 'high': 60}],
                          fitness_func=gdc_fitness_func,
                          parent_selection_type=parent_selection_type,
                          mutation_type=mutation_type,
                          mutation_num_genes=mutation_num_genes,
                          # on_generation = callback_generation,
                          )

  # 유전자 알고리즘 실행
  print("GDC 유전자 알고리즘 실행 ......")
  ga_instance.run()

  best_solution = ga_instance.best_solution() #전체 최적 결과(현재 까지 찾은 최적 해와 적합도중 best를 출력)
  best_period_short = best_solution[0][0]
  best_period_long = best_solution[0][1]

  return best_period_short, best_period_long

#-------------------------------------ga 바탕 rsi signal----------------------------------------------
# 시그널 생성 함수
def make_gdc_ga_signal(df, best_period_short, best_period_long):
  # 단기와 장기 이동평균을 계산
  df['ga_short_ma'] = talib.SMA(df['Close'], timeperiod=best_period_short)
  df['ga_long_ma'] = talib.SMA(df['Close'], timeperiod=best_period_long)

  # 시그널 생성
  df['ga_gdc_signal'] = 0 # 초기 시그널 값을 0으로 설정

  # 단기 이동평균선이 중장기 이동평균선을 상향돌파  => 매수시점(1)
  # 전일 : 단기이동평균선 < 중장기 이동평균선 / 후일(현재 지점) : 단기이동평균선 > 중장기 이동평균선
  df.loc[(df['ga_short_ma'].shift() < df['ga_long_ma'].shift()) & (df['ga_short_ma'] > df['ga_long_ma']), 'ga_gdc_signal'] = 1

  # 단기 이동평균선이 중장기 이동평균선을 하향돌파  => 매도시점(-1)
  # 전일 : 단기이동평균선 > 중장기 이동평균선 / 후일(현재 지점) : 단기이동평균선 < 중장기 이동평균선
  df.loc[(df['ga_short_ma'].shift() > df['ga_long_ma'].shift()) & (df['ga_short_ma'] < df['ga_long_ma']), 'ga_gdc_signal'] = -1

  return df['ga_gdc_signal']


#-------------------------------최고 수익률을 내는 임계값-----------------------------
def best_gdc_profit(df, stock, type):
      # best_period_short, best_period_long = 9.64751428202647, 58.25296883457129 # AAPL
      # best_period_short, best_period_long = 6.742119244332138, 35.5828278595837  # GOOGL
      # best_period_short, best_period_long = 6.04215355561664. 32.97685207142818  # IONQ
      # best_period_short, best_period_long = 9.817821838643662, 49.03616693501824  # MSFT
      # best_period_short, best_period_long = 3.4940058371053206, 35.722332055387746  # NVDA
      # best_period_short, best_period_long = 9.717448550407331, 49.129268909554064 # RKLB
      # best_period_short, best_period_long = 4.624832,  32.208787 # TSM

  if type == 'GA' or type == 'ALL':
      # 최적화된 매개변수 출력 : ESN 최적화한 상태에서 10번 돌렸을 때 가장 높은 profit이 나온 임계치를 저장함.
      parameters = get_param(stock, 'gdc')
      # print(f"parameters : {parameters}")
      best_period_short, best_period_long = parameters[2], parameters[3]
      # print(f"best_period_short, best_period_long : {best_period_short}, {best_period_long}")

  elif type == 'ESN'  or type == 'NO':
      # 매개변수 최적화하기
      best_period_short, best_period_long = ga_gdc_optimize(df)

  # 수익률 확인 및 최적 매개변수로 시그널 갱신
  make_gdc_ga_signal(df, best_period_short, best_period_long)
  fit_returns = backtesting(df, 'gdc')[0]

  return best_period_short, best_period_long, fit_returns, df



# 최종적으로 모듈화된 수익률 도출 함수
def gdc_test(df, stock, type):
  # train 데이터에 기본 시그널 생성
  make_gdc_sig(df, 'short_ma', 'long_ma', 5, 120, 'gdc_signal')

  # GA로 필요한 매개변수 최적화
  gdc_short, gdc_long, gdc_fit, gdc_df = best_gdc_profit(df, stock, type)

  return gdc_df['ga_gdc_signal'], gdc_short, gdc_long, gdc_fit
