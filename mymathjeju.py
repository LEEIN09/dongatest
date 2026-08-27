import os
import sys
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage

# ── Windows 콘솔 UTF-8 인코딩 설정 ───────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ── 1. 환경 변수 로드 (.env) ──────────────────────────────────
load_dotenv()


# ── 2. Pydantic BaseModel 입력 스키마 정의 ──────────────────
class MathQuery(BaseModel):
    a: float = Field(..., description="첫 번째 숫자")
    b: float = Field(..., description="두 번째 숫자")
    operation: str = Field(
        ..., 
        description="수행할 사칙연산 종류 ('add', 'subtract', 'multiply', 'divide', 'abs')"
    )


class WeatherQuery(BaseModel):
    location: str = Field(..., description="날씨를 조회할 지역 또는 도시명 (예: 제주, 서울, 부산, 도쿄, 뉴욕)")
    unit: str = Field(
        default="celsius", 
        description="온도 단위 ('celsius' 또는 'fahrenheit')"
    )


# ── (신규) 관심사항(영화) 스키마 클래스: Myfavority ─────────
class Myfavority(BaseModel):
    genre: str = Field(
        default="전체", 
        description="영화 장르 (예: 액션, SF, 애니메이션, 드라마, 스릴러, 전체)"
    )
    time_slot: str = Field(
        default="오후", 
        description="하루 3번 정보 체크 시간대 ('아침' - 조조 및 신작 개봉작, '오후' - 실시간 박스오피스 순위, '저녁' - 퇴근/심야 추천 영화)"
    )


# ── 3. LangChain 도구(@tool) 정의 ───────────────────────────
@tool(args_schema=MathQuery)
def calculate_math(a: float, b: float, operation: str) -> str:
    """ 두 수에 대한 사칙연산(더하기, 빼기, 곱하기, 나누기) 및 절댓값을 수행하는 함수 """
    op = operation.lower().strip()
    if op in ["add", "더하기", "+"]:
        return f"{a} + {b} = {a + b}"
    elif op in ["subtract", "빼기", "-"]:
        return f"{a} - {b} = {a - b}"
    elif op in ["multiply", "곱하기", "*"]:
        return f"{a} * {b} = {a * b}"
    elif op in ["divide", "나누기", "/"]:
        if b == 0:
            return "❌ 오류: 0으로 나눌 수 없습니다."
        return f"{a} / {b} = {a / b}"
    elif op in ["abs", "절댓값"]:
        diff = a - b
        return f"|{a} - {b}| = |{diff}| = {abs(diff)}"
    else:
        return f"❌ 지원하지 않는 연산자입니다: {operation}"


@tool(args_schema=WeatherQuery)
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """ 지정된 지역(제주도, 서울 등)의 현재 날씨와 기온 정보를 조회하는 함수 """
    weather_data = {
        "제주": {"temp": "27°C", "condition": "맑고 시원한 바람 🍊", "humidity": "60%", "tip": "해안 도로 드라이브나 한라산 탐방하기 좋은 날씨입니다!"},
        "제주도": {"temp": "27°C", "condition": "맑고 시원한 바람 🍊", "humidity": "60%", "tip": "해안 도로 드라이브나 한라산 탐방하기 좋은 날씨입니다!"},
        "서울": {"temp": "24°C", "condition": "맑음 ☀️", "humidity": "50%", "tip": "나들이하기 좋은 쾌청한 날씨입니다."},
        "부산": {"temp": "25°C", "condition": "맑음 ☀️", "humidity": "58%", "tip": "해운대 해변 산책하기 좋습니다."},
        "도쿄": {"temp": "26°C", "condition": "흐림 ☁️", "humidity": "65%", "tip": "우산을 챙기시면 좋습니다."},
        "뉴욕": {"temp": "18°C", "condition": "비 🌧️", "humidity": "80%", "tip": "비가 내리니 실내 활동을 권장합니다."},
    }
    
    info = weather_data.get(location, {
        "temp": "23°C", 
        "condition": "쾌청 🌤️", 
        "humidity": "55%", 
        "tip": "기분 좋은 바람이 부는 날씨입니다."
    })
    
    return f"[{location}] 날씨: {info['condition']}, 기온: {info['temp']} (단위: {unit}), 습도: {info['humidity']} (Tip: {info['tip']})"


@tool(args_schema=Myfavority)
def check_movie_info(genre: str = "전체", time_slot: str = "오후") -> str:
    """ 하루 3번(아침, 오후, 저녁) 관심 영화 정보 및 박스오피스를 체크하는 함수 """
    time_slot = time_slot.strip()
    
    movie_schedule = {
        "아침": {
            "title": "🌅 [아침 체크] 오늘의 신작 개봉 & 조조 상영작 안내",
            "movies": [
                "1. 인터스텔라 재개봉 (SF/우주) - 조조할인 상영 중",
                "2. 듄: 파트2 (SF/액션) - IMAX 특별관 상영",
                "3. 인사이드 아웃 2 (애니메이션) - 가족 관람 추천"
            ]
        },
        "오후": {
            "title": "☀️ [오후 체크] 실시간 박스오피스 TOP 3 순위 & 예매율",
            "movies": [
                "🥇 1위: 파묘 (미스터리/스릴러) - 예매율 38.5%",
                "🥈 2위: 범죄도시 4 (액션/범죄) - 예매율 27.2%",
                "🥉 3위: 쿵푸팬더 4 (애니메이션/코믹) - 예매율 15.8%"
            ]
        },
        "저녁": {
            "title": "🌙 [저녁 체크] 퇴근 후 심야 영화 & OTT 명작 추천",
            "movies": [
                "🎬 극장 추천: 오펜하이머 (드라마/전기) - 몰입감 최고",
                "📺 OTT 추천: 라라랜드 (뮤지컬/로맨스) - 힐링 저녁 영화",
                "🍿 심야 추천: 다크나이트 (액션/느와르) - 스트레스 해소"
            ]
        }
    }
    
    selected = movie_schedule.get(time_slot, movie_schedule["오후"])
    movie_list = "\n   ".join(selected["movies"])
    
    return (
        f"{selected['title']}\n"
        f"   선택 장르: [{genre}]\n"
        f"   {movie_list}\n"
        f"   💡 Tip: 하루 3번(아침 09시, 오후 14시, 저녁 20시) 업데이트됩니다."
    )


tool_dict = {
    "calculate_math": calculate_math,
    "get_current_weather": get_current_weather,
    "check_movie_info": check_movie_info
}
tools = [calculate_math, get_current_weather, check_movie_info]


# ── 4. OpenRouter API 모델 설정 및 도구 바인딩 ───────────────
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

llm_with_tools = model.bind_tools(tools)


# ── 5. 도구 실행 및 요약 핸들러 (RunnableLambda 함수) ─────────
def execute_tools_and_summarize(ai_message):
    """ 도구 호출(tool_calls)이 있으면 실행하고 결과를 모델에 전달하여 최종 답변 생성 """
    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        tool_messages = []
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"🛠️  [도구 호출]: {tool_name}")
            print(f"📌  [전달 인자]: {tool_args}")

            selected_tool = tool_dict.get(tool_name)
            if selected_tool:
                tool_msg = selected_tool.invoke(tool_call)
                tool_messages.append(tool_msg)
                print(f"📊  [실행 결과]: {tool_msg.content}\n")

        # 도구 실행 결과를 바탕으로 2차 답변 생성
        final_prompt_messages = [
            SystemMessage("당신은 전달된 수학/날씨/영화 도구 실행 결과를 바탕으로 친절하고 자연스러운 한국어로 최종 답변을 작성하는 AI 어시스턴트입니다."),
            ai_message,
            *tool_messages
        ]
        return model.invoke(final_prompt_messages)
    
    return ai_message


# ── 6. LangChain LCEL 체인 구성 (| 파이프 연산자) ────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 사용자의 질문에 답변하기 위해 수학(calculate_math), 날씨(get_current_weather), 영화 정보(check_movie_info) 도구를 적극적으로 활용하는 유능한 AI 어시스턴트입니다."),
    ("user", "{question}")
])

# [프롬프트] | [도구 바인딩 LLM] | [도구 실행 및 요약] | [문자열 출력 파서]
chain = (
    prompt
    | llm_with_tools
    | RunnableLambda(execute_tools_and_summarize)
    | StrOutputParser()
)


# ── 7. 실행 함수 ─────────────────────────────────────────────
def ask(query: str):
    print("=" * 60)
    print(f"👤 사용자 질문: {query}")
    print("=" * 60)
    
    response = chain.invoke({"question": query})
    
    print("🤖 최종 AI 답변:")
    print(response)
    print("\n")


# ── 8. 메인 실행 ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🎬 수학 & 제주 날씨 & 영화 관심사 도구 에이전트 (mymathjeju.py)")
    print("=" * 60 + "\n")

    # 예제 1: 관심사항 영화 정보 체크 (저녁 시간대 추천)
    ask("오늘 저녁에 볼만한 영화 추천 정보 체크해줘!")

    # 예제 2: 관심사항 영화 정보 체크 (오후 실시간 박스오피스)
    ask("지금 오후 실시간 영화 박스오피스 순위 알려줘.")

    # 예제 3: 제주도 날씨 질의
    ask("제주도의 현재 날씨와 여행 팁 알려줘!")

    # 예제 4: 수학 연산 질의
    ask("125와 35를 곱하면 얼마야?")
