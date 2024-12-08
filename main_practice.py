from fastapi import FastAPI

from pydantic import BaseModel
# 터미널 : uvicorn main_run:app --reload

#
# '''FastAPI 애플리케이션을 만들고 테스트해보는 과정 코드'''
# # FastAPI 앱 객체 만들기
# app = FastAPI()
#
#
# '''데이터 유효성 검사 및 설정 관리'''
# class Item(BaseModel):
#     name: str
#     description: str = None
#     price: float
#     tax: float = None
#
#
# @app.post("/items/")
# async def create_item(item: Item):
#     return item
#
#
#
#
# # 루트 경로로 get 요청
# @app.get("/")
# async def read_root():
#     return {"Hello":"World"} #Json형식으로 응답
#
# # 경로상에서 값을 추출하겠다 {}
# @app.get("/items/{item_id}")
# async def read_item(item_id: int, q : str = None):
#     # 매개변수 : item_id(경로매개변수, type: int)
#     # / 쿼리 매개변수 q : (type:string) (기본값 : None)
#     return {'item_id' : item_id, 'q' : q}
#
#
#
# # 크롬 주소창에 http://127.0.0.1:8000/items/100?q=hello, 이런 식으로 입력 가능 (자료형 안 지키면 오류 뜸 ㅎㅎ)


