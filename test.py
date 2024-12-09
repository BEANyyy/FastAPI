import requests

# url = "http://localhost:8088/detect"
# message = "Test message"
# file_path = "sample.jpg"
#
# with open(file_path, "rb") as file:
#     response = requests.post(url, data={"message": message},
#                              files={"file": file})
#
#
# print(response.json())

#
# url = "http://127.0.0.1:8088/today"
# message = "Test message"
# stock_name = "AMZN"
# print(f"stock_name : {stock_name}")
# print("모델을 돌리는 중입니다. 잠시만 기다려주세요. . . . . . . .")
#
# response = requests.post(url, data={"message": message, "stock_name": stock_name})


url = "http://127.0.0.1:8088/realtime"
message = "Test message"
stock_name = "AAPL"
close = 255
print(f"stock_name : {stock_name}")
print("모델을 돌리는 중입니다. 잠시만 기다려주세요. . . . . . . .")

response = requests.post(url, data={"message": message, "stock_name": stock_name, "close": close})
print(response.json())