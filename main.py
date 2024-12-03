from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import io
import base64
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
from stock import predict_model

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()


# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')


# 데이터 모델 정의
class DetectionResult(BaseModel):
    message: str
    image: str

class PredictionResult(BaseModel):
    message : str
    signal : int



# 객체 탐지 함수
def detect_objects(image: Image):
    img = np.array(image)    # 이미지를 numpy 배열로 변환
    results = model(img)    # 객체 탐지
    class_names = model.names   # 클래스이름 저장

    # 결과를 바운딩 박스, 클래스 이름, 정확도로 이미지에 표시
    for result in results:
        boxes = result.boxes.xyxy # 바운딩 박스 : xy 점 두 개에 대한 정보를 반환
        confidences = result.boxes.conf # 신뢰도
        class_ids = result.boxes.cls    # 클래스 이름
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)  # 좌표를 정수로 변환
            label = class_names[int(class_id)]  # 클래스 이름
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    result_image = Image.fromarray(img)
    return result_image # 결과 이미지를 PIL로 변환


# 기본 엔드 포인트
@app.get("/")
async def index():
    return {"message": "Hello FastAPI"}


# 객체 탐지 엔드 포인트
@app.post("/predict", response_model=PredictionResult)
async def predict_service(message: str = Form(...), stock_name: str = Form(...)):
    # 모델 실행 및 로그 수집
    result_signal = predict_model(stock_name)
    print(result_signal)
    return PredictionResult(message=message, signal=result_signal)


# 객체 탐지 엔드 포인트
@app.post("/detect", response_model=DetectionResult)
async def detect_service(message: str = Form(...), file: UploadFile = File(...)):
    # 이미지를 읽어서 PIL 이미지로 변환
    image = Image.open(io.BytesIO(await file.read()))

    # 알파 채널 제거하고 RGB로 변환
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # 객체 탐지 수행
    result_image = detect_objects(image)

    # 이미지 결과를 base64로 인코딩
    buffered = io.BytesIO()
    result_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return DetectionResult(message=message, image=img_str)


# 애플리케이션 실행을 위한 정의
#파이썬명령으로 바로 실행할 때 필요한 코드, uvicorn은 없어도 상관 없음.
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)