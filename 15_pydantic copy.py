 #pydantic 
import yfinance as yf 
import pytz

from datetime import datetime
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv()
# model = ChatOpenAI(model="gpt-4o-mini",  temperature=0)
model = ChatOpenAI(model="gpt-4o")


from pydantic import BaseModel, Field

class StockHistoryInput(BaseModel):   
    ticker: str = Field(..., title='주식코드', description='주식 코드 (예: AAPL, TSLA)')
    period: str = Field(..., title='기간', description='주식 데이터 조회 기간 (예: 1d, 1mo, 1y)')


@tool(args_schema=StockHistoryInput)
def get_yf_stock_history(ticker: str, period: str) -> str:
    """ 주식 종목의 가격 데이터를 조회하는 함수 """
    stock = yf.Ticker(ticker)  # class Ticker(TickerBase): 클래스
    history = stock.history(period=period)
    print('history타입', type(history)) # history타입 <class 'pandas.DataFrame'>
    print()
    history_md = history.to_markdown()
    return history_md

# to_markdown() 함수는 파이썬의 데이터 분석 라이브러리인 Pandas(판다스)에서 제공하는 함수로, 
# yfinance가 반환한 표 형태의 주식 데이터(DataFrame)를 마크다운(Markdown) 형식의 문자열 표로 변환하는 역할

@tool
def get_current_time(timezone: str, location: str) -> str:
    """
        현재 시간을 YYYY-MM-DD HH:MI:SS 형식으로 반환하는 함수
       
        Args:
            timezone (str): 타임존(예: "Asia/Seoul"). 실제 존재해야 함.
            location (str): 지역명. 타임존은 모든 지역에 대응하지 않으며, 이후 llm 답변 생성에 사용됨. (함수의 메타데이터 만들어줌)
    """
    tz = pytz.timezone(timezone)
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    return f"{timezone} ({location}) 현재시간 {now}"


# 랭체인 @tool포함된 함수 기능처럼 이름으로 접근 가능 
tools = [get_yf_stock_history, get_current_time]
tool_dict = {
    "get_current_time": get_current_time,
    "get_yf_stock_history": get_yf_stock_history
}

# 도구를 모델에 연결 바인딩  openrouter/올라마 나중에 연결
llm_with_tools = model.bind_tools(tools)
# print('llm_with_tools 결과')
# print(llm_with_tools)

# 도구를 사용해 언어 모델 답변 생성
messages = [
    SystemMessage("당신은 사용자의 질문에 답변을 하기 위해 tools를 사용할 수 있다."),
    HumanMessage("테슬라의 최근 3일간 주가 정보는 어떻게 되지?")
]

response = llm_with_tools.invoke(messages)
messages.append(response)


for tool_call in response.tool_calls:
    selected_tool = tool_dict.get(tool_call['name'])
    if selected_tool:
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)


response = llm_with_tools.invoke(messages)
print(response)
print('- ' * 50)
print()
print(response.content)


# ModuleNotFoundError: No module named 'yfinance' 
# pip install yfinance

# ModuleNotFoundError: No module named 'pytz'
# pip install pytz

# ImportError: `Import tabulate` failed.  Use pip or conda to install the tabulate package.
# pip install tabulate

