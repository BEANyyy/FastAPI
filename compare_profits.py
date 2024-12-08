from kibon import backtesting, new_backtesting, show_graph

import pandas as pd


def compare_profits(stock, type):
    df = pd.read_csv(f'STOCK/{stock}/{type}.csv')
    print(df)


    result = backtesting(df, 'result')
    # result = new_backtesting(df, 'result')
    print("result : ", result)

    # show_graph(df, 'RESULT', result[1], result[2])

    return result[0]














