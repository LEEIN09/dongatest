import os
import sys
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage

# ── 환경 변수 로드 ──────────────────────────────────────────
load_dotenv()

# ── Streamlit 페이지 설정 ───────────────────────────────────
st.set_page_config(
    page_title="AI 에이전트 도구 시스템 (LCEL)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 1. st.session_state 초기화 ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "tool_usage_stats" not in st.session_state:
    st.session_state.tool_usage_stats = {
        "calculate_math": 0,
        "get_current_weather": 0
    }


# ── 2. Pydantic BaseModel 스키마 정의 ───────────────────────
class MathQuery(BaseModel):
    a: float = Field(..., description="첫 번째 숫자")
    b: float = Field(..., description="두 번째 숫자")
    operation: str = Field(
        ..., 
        description="수행할 사칙연산 종류 ('add', 'subtract', 'multiply', 'divide')"
    )


class WeatherQuery(BaseModel):
    location: str = Field(..., description="날씨를 조회할 도시 또는 지역명 (예: 서울, 도쿄, 뉴욕)")
    unit: str = Field(
        default="celsius", 
        description="온도 단위 ('celsius' 또는 'fahrenheit')"
    )


# ── 3. LangChain 도구(@tool) 정의 ───────────────────────────
@tool(args_schema=MathQuery)
def calculate_math(a: float, b: float, operation: str) -> str:
    """ 두 수에 대한 사칙연산(더하기, 빼기, 곱하기, 나누기)을 수행하는 함수 """
    op = operation.lower()
    if op in ["add", "더하기", "+"]:
        return f"{a} + {b} = {a + b}"
    elif op in ["subtract", "빼기", "-"]:
        return f"{a} - {b} = {a - b}"
    elif op in ["multiply", "곱하기", "*"]:
        return f"{a} * {b} = {a * b}"
    elif op in ["divide", "나누기", "/"]:
        if b == 0:
            return "오류: 0으로 나눌 수 없습니다."
        return f"{a} / {b} = {a / b}"
    else:
        return f"지원하지 않는 연산자입니다: {operation}"


@tool(args_schema=WeatherQuery)
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """ 지정된 지역의 현재 날씨와 기온 정보를 조회하는 함수 """
    weather_data = {
        "서울": {"temp": "24°C", "condition": "맑음 ☀️", "humidity": "50%"},
        "도쿄": {"temp": "26°C", "condition": "흐림 ☁️", "humidity": "65%"},
        "뉴욕": {"temp": "18°C", "condition": "비 🌧️", "humidity": "80%"},
        "부산": {"temp": "25°C", "condition": "맑음 ☀️", "humidity": "58%"},
        "제주": {"temp": "27°C", "condition": "바람 💨", "humidity": "60%"},
    }
    info = weather_data.get(location, {"temp": "22°C", "condition": "쾌청 🌤️", "humidity": "55%"})
    return f"[{location}] 날씨: {info['condition']}, 기온: {info['temp']} (단위: {unit}), 습도: {info['humidity']}"


tool_dict = {
    "calculate_math": calculate_math,
    "get_current_weather": get_current_weather
}
tools = [calculate_math, get_current_weather]


# ── 4. 사이드바(Sidebar) UI 및 세션 관리 ────────────────────
with st.sidebar:
    st.title("⚙️ 에이전트 설정")
    st.markdown("---")

    # API 키 설정
    default_api_key = os.getenv("OPENROUTER_API_KEY", "")
    api_key_input = st.text_input(
        "🔑 OpenRouter API Key",
        value=default_api_key,
        type="password",
        help=".env 파일에서 불러오거나 직접 입력할 수 있습니다."
    )

    # 모델 선택
    selected_model = st.selectbox(
        "🧠 LLM 모델 선택",
        options=[
            "openai/gpt-4o-mini",
            "google/gemini-2.5-flash",
            "deepseek/deepseek-chat",
            "meta-llama/llama-3.3-70b-instruct",
            "openai/gpt-4o"
        ],
        index=0
    )

    # 온도(Temperature) 슬라이더
    temperature = st.slider("🌡️ Temperature (창의성)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    st.markdown("---")
    # 세션 통계 표시
    st.subheader("📊 세션 통계 (st.session_state)")
    st.metric("총 질의 횟수", f"{st.session_state.query_count} 회")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("수학 도구", f"{st.session_state.tool_usage_stats['calculate_math']} 회")
    with col_s2:
        st.metric("날씨 도구", f"{st.session_state.tool_usage_stats['get_current_weather']} 회")

    st.markdown("---")
    st.subheader("🛠️ 등록된 도구 목록")
    st.info(
        "**1. MathQuery (`calculate_math`)**\n"
        "- 사칙연산(더하기, 빼기, 곱하기, 나누기)\n\n"
        "**2. WeatherQuery (`get_current_weather`)**\n"
        "- 지역별 날씨 및 기온 조회"
    )

    # 대화 기록 다운로드 및 초기화 버튼
    if st.session_state.messages:
        chat_log = "\n\n".join([f"[{m['role'].upper()} - {m.get('time', '')}]\n{m['content']}" for m in st.session_state.messages])
        st.download_button(
            label="💾 대화 기록 다운로드",
            data=chat_log,
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )

    if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.session_state.tool_usage_stats = {"calculate_math": 0, "get_current_weather": 0}
        st.rerun()


# ── 5. 메인 화면 UI 및 LCEL 처리 ─────────────────────────────
st.title("🤖 LangChain LCEL 도구 에이전트")
st.caption("Pydantic 스키마(MathQuery, WeatherQuery) 기반의 도구 호출 및 LCEL 파이프라인")

# 이전 대화 메시지 출력 (st.session_state.messages 기반)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "tool_info" in msg and msg["tool_info"]:
            with st.expander("🔍 도구 실행 세부 정보", expanded=False):
                st.code(msg["tool_info"], language="text")

# 빠른 예시 질문 버튼
col1, col2, col3 = st.columns(3)
quick_query = None
if col1.button("📍 서울의 현재 날씨는?"):
    quick_query = "서울의 현재 날씨는 어때?"
if col2.button("🔢 125 곱하기 35는?"):
    quick_query = "125와 35를 곱하면 얼마야?"
if col3.button("🌤️ 도쿄 날씨 & 48 나누기 6"):
    quick_query = "도쿄의 날씨를 알려주고, 48을 6으로 나눈 값도 계산해줘."

# 사용자 입력 받기
user_input = st.chat_input("질문을 입력하세요 (예: 서울 날씨 알려줘, 450 + 120 계산해줘)...")

query_to_run = user_input or quick_query

if query_to_run:
    if not api_key_input:
        st.error("⚠️ OpenRouter API Key를 사이드바에 입력하거나 .env 파일에 설정해 주세요.")
        st.stop()

    now_str = datetime.now().strftime("%H:%M:%S")

    # 세션에 사용자 메시지 추가
    st.session_state.messages.append({
        "role": "user", 
        "content": query_to_run,
        "time": now_str
    })
    st.session_state.query_count += 1

    with st.chat_message("user"):
        st.markdown(query_to_run)

    # ── LCEL 파이프라인 구성 ──
    model = ChatOpenAI(
        model=selected_model,
        api_key=api_key_input,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature
    )
    llm_with_tools = model.bind_tools(tools)

    tool_logs = []

    def execute_tools_and_summarize(ai_message):
        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            tool_messages = []
            for tool_call in ai_message.tool_calls:
                t_name = tool_call["name"]
                t_args = tool_call["args"]
                tool_logs.append(f"🛠️ 도구: {t_name}\n📌 인자: {t_args}")
                
                # 도구 사용 횟수 통계 업데이트
                if t_name in st.session_state.tool_usage_stats:
                    st.session_state.tool_usage_stats[t_name] += 1

                selected_tool = tool_dict.get(t_name)
                if selected_tool:
                    tool_msg = selected_tool.invoke(tool_call)
                    tool_messages.append(tool_msg)
                    tool_logs.append(f"📊 결과: {tool_msg.content}\n" + "-"*40)
            
            final_messages = [
                SystemMessage("당신은 전달된 도구 실행 결과를 바탕으로 친절하고 자연스러운 한국어로 최종 답변을 작성하는 AI 어시스턴트입니다."),
                ai_message,
                *tool_messages
            ]
            return model.invoke(final_messages)
        return ai_message

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 사용자의 질문에 답변하기 위해 수학 및 날씨 도구를 적절히 활용하는 AI 어시스턴트입니다."),
        ("user", "{question}")
    ])

    chain = (
        prompt 
        | llm_with_tools 
        | RunnableLambda(execute_tools_and_summarize) 
        | StrOutputParser()
    )

    # 답변 생성 및 출력
    with st.chat_message("assistant"):
        with st.spinner("도구 분석 및 답변 생성 중..."):
            try:
                final_answer = chain.invoke({"question": query_to_run})
                st.markdown(final_answer)

                tool_info_str = "\n".join(tool_logs) if tool_logs else ""
                if tool_info_str:
                    with st.expander("🔍 도구 실행 세부 정보", expanded=False):
                        st.code(tool_info_str, language="text")

                # 세션에 어시스턴트 메시지 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "tool_info": tool_info_str,
                    "time": datetime.now().strftime("%H:%M:%S")
                })
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
