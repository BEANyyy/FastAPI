from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import io
import base64
from PIL import Image
from matplotlib.figure import Figure
from ultralytics import YOLO
from kibon import encode_figure_to_base64

from stock import realtime_predict, today_predict

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()


# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')


# 데이터 모델 정의
class DetectionResult(BaseModel):
    message: str
    image: str

class TodayPredictionResult(BaseModel):
    message : str
    signal : int
    suggestion : str
    date : str
    graph: str  # 그래프 이미지의 Base64 인코딩 데이터

class RealtimePredictionResult(BaseModel):
    message : str
    signal : int
    suggestion : str
    graph: str  # 그래프 이미지의 Base64 인코딩 데이터



# 기본 엔드 포인트
@app.get("/")
async def index():
    return {"message": "Hello FastAPI"}

# 실시간 분석
@app.post("/realtime", response_model=RealtimePredictionResult)
async def realtime_service(
    message: str = Form(...),
    stock_name: str = Form(...),
    close: float = Form(...)
):
    # 모델 실행 및 로그 수집
    signal_result, graph = realtime_predict(stock_name, close)
    print(signal_result)

    if signal_result == -1:
        suggestion = "파세요"
    elif signal_result == 1:
        suggestion = "사세요"
        print("오늘의 거래 제안 : 사세요")
    else:
        suggestion = "현 상태를 유지하세요"

    # 그래프를 Base64로 변환
    graph_base64 = encode_figure_to_base64(graph) if graph else ""

    return RealtimePredictionResult(message=message, suggestion=suggestion, signal=signal_result, graph=graph_base64)

# 오늘의 시그널
@app.post("/today", response_model=TodayPredictionResult)
async def today_predict_service(
        message: str = Form(...),
        stock_name: str = Form(...)
):
    # 모델 실행 및 로그 수집
    signal_result, date, graph = today_predict(stock_name)
    print(signal_result)

    if signal_result == -1:
        suggestion = "파세요"
    elif signal_result== 1:
        suggestion = "사세요"
        print("오늘의 거래 제안 : 사세요")
    else:

        suggestion = "현 상태를 유지하세요"

    # 그래프를 Base64로 변환
    graph_base64 = encode_figure_to_base64(graph) if graph else ""


    return TodayPredictionResult(message=message, suggestion=suggestion, signal=signal_result, date=date, graph=graph_base64)


# # 객체 탐지 엔드 포인트
# @app.post("/detect", response_model=DetectionResult)
# async def detect_service(message: str = Form(...), file: UploadFile = File(...)):
#     # 이미지를 읽어서 PIL 이미지로 변환
#     image = Image.open(io.BytesIO(await file.read()))
#
#     # 알파 채널 제거하고 RGB로 변환
#     if image.mode == 'RGBA':
#         image = image.convert('RGB')
#     elif image.mode != 'RGB':
#         image = image.convert('RGB')
#
#     # 객체 탐지 수행
#     result_image = detect_objects(image)
#
#     # 이미지 결과를 base64로 인코딩
#     buffered = io.BytesIO()
#     result_image.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#
#     return DetectionResult(message=message, image=img_str)


# 애플리케이션 실행을 위한 정의
#파이썬명령으로 바로 실행할 때 필요한 코드, uvicorn은 없어도 상관 없음.
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)