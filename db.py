import pymysql
from datetime import datetime
import os
import pandas as pd

# MySQL 데이터베이스 연결
def connect_to_db(host, user, password, database):
    try:
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        print("MySQL 데이터베이스에 성공적으로 연결되었습니다!")
        return connection
    except Exception as e:
        print(f"오류 발생 : {e}")
        return None

# 데이터 삽입 함수
def insert_data(connection, table_name, data):
    try:
        cursor = connection.cursor()
        # 컬럼명과 데이터를 동적으로 설정
        placeholders = ', '.join(['%s'] * len(data))
        columns = ', '.join(data.keys())
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, list(data.values()))
        connection.commit()
        print(f"데이터 삽입 성공: {data}")
    except Exception as e:
        print(f"데이터 삽입 중 오류 발생: {e}")
    finally:
        cursor.close()


# 조건 값으로 데이터 조회 함수
def fetch_data(connection, table_name, column_name, condition_value):
    try:
        cursor = connection.cursor()
        sql = f"SELECT * FROM {table_name} WHERE {column_name} = %s"  # 조건을 동적으로 추가
        cursor.execute(sql, (condition_value,))  # 값은 튜플로 전달
        result = cursor.fetchall()
        print(f"조건에 맞는 데이터 조회 성공! ({column_name} = {condition_value})")
        return result
    except pymysql.MySQLError as e:
        print(f"데이터 조회 중 오류 발생: {e}")
        return None
    finally:
        cursor.close()


def get_param(stock, type):
    # 혹시몰라서적어두는최적화된파라미터,,,
    # # best_sparsity, best_rho, best_noise = 0.4, 1.7, 0.1  # NVDA
    # # best_sparsity, best_rho, best_noise = 0.4, 1.3, 0.1  # NVDY
    # # best_sparsity, best_rho, best_noise = 0.4, 0.9, 0.1  # AAPL
    # # best_sparsity, best_rho, best_noise = 0.1, 1.3, 0.01  # TSLA
    # # best_sparsity, best_rho, best_noise = 0.1, 1.7, 0.1  # MSFT
    # # best_sparsity, best_rho, best_noise = 0.1, 0.9, 0.01   # TSLL
    # # best_sparsity, best_rho, best_noise = 0.1, 0.9, 0.1  # GOOGL
    # # best_sparsity, best_rho, best_noise = 0.3, 0.9, 0.1  # NVDL
    # # best_sparsity, best_rho, best_noise = 0.4, 1.3, 0.01  # RKLB

    result_param = None

    print(__name__)
    print("---------------db---------------")
    # 메인 실행
    if __name__ == "db":
    # if __name__ == "__main__":
        # 연결 정보 설정
        host = "localhost"         # MySQL 서버 호스트 (로컬의 경우 localhost)
        user = "root"              # MySQL 사용자 이름
        password = "1234"   # MySQL 비밀번호
        database = "stock_db"       # 데이터베이스 이름

        # 데이터베이스 연결
        connection = connect_to_db(host, user, password, database)


        if connection:
            # 조건 값을 사용하여 데이터 가져오기
            table_name = type  # 조회할 테이블 이름
            column_name = "stock_id"  # 조건을 적용할 컬럼 이름
            condition_value = stock  # 조건 값

            data = fetch_data(connection, table_name, column_name, condition_value)


            # 가져온 데이터 출력
            if data:
                # 날짜 기준으로 정렬
                sorted_data = sorted(data, key=lambda x: x[-1], reverse=True)  # reverse=True : 가장 최신 값 / reverse=True : 가장 옛날 값 출력됨

                # 가장 최신 데이터
                latest_data = sorted_data[0]

                print("------------------------")
                print(f"현재 {type} param의 생성 기준 날짜")
                print(latest_data[-1])
                print("------------------------")

                result_param = latest_data
                print(result_param)
                # esn_param = latest_data
                # best_sparsity, best_rho, best_noise = esn_param[2], esn_param[3], esn_param[4]
                #
                # print(f"best_sparsity, best_rho, best_noise = {best_sparsity}, {best_rho}, {best_noise}")

    print("--------------------------------")
    # return best_sparsity, best_rho, best_noise
    return result_param



def upload_signals_to_db(csv_path, db_config):
    # Extract stock_name and model_name from the csv_path
    path_parts = csv_path.split('/')
    stock_name = path_parts[-2]  # Second to last part of the path
    print(f"stock_name : {stock_name}")
    model_name = os.path.splitext(path_parts[-1])[0]  # File name without extension
    print(f"model_name : {model_name}")

    # Load the CSV file
    data = pd.read_csv(csv_path).tail(1) # 가장 최근 데이터만 업로드하도록 함.

    # Add additional columns
    today_date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print(f"today_date : {today_date}")
    data['stock_name'] = stock_name
    data['model_name'] = model_name
    data['update_date'] = today_date

    # Rename columns to match the database schema
    data = data.rename(columns={
        'Date': 'date',
        'Close': 'close',
        'pred_sig': 'signal'
    })

    # Reorder columns to match the database schema
    data = data[['stock_name', 'model_name', 'date', 'close', 'signal', 'update_date']]

    # Connect to the MySQL database
    try:
        conn = pymysql.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        cursor = conn.cursor()

        # Insert the data into the database
        for _, row in data.iterrows():
            sql = '''INSERT INTO signals (stock_name, model_name, date, close, `signal`, update_date)
                     VALUES (%s, %s, %s, %s, %s, %s)'''
            cursor.execute(sql, tuple(row))

        conn.commit()
        print("Data uploaded successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# parameters = get_param('AAPL', 'esn')
# print(f"parameters : {parameters}")
# best_sparsity, best_rho, best_noise = parameters[2], parameters[3], parameters[4]
# print(f"best_sparsity, best_rho, best_noise : {best_sparsity}, {best_rho}, {best_noise}")


# stock_list = ['AAPL', 'GOOGL', 'IONQ', 'MSFT', 'NVDA', 'RKLB', 'TSM']
#
# for stock in stock_list:
#     parameters = get_param(stock, 'stoch')
#     print(f"parameters : {parameters}")
#     best_period_K, best_period_D = parameters[2], parameters[3]
#     print(f"best_period_K, best_period_D : {best_period_K}, {best_period_D}")




