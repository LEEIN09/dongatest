import os
import sys
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# ── Windows 콘솔 UTF-8 인코딩 설정 ───────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ── 환경 변수 로드 ──────────────────────────────────────────
load_dotenv()

# ── Streamlit 페이지 설정 ───────────────────────────────────
st.set_page_config(
    page_title="AI 에이전트 시스템 (test2.py)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 1. st.session_state 세션 상태 초기화 ────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0


# ── 2. Pydantic 입력 스키마 (MathQuery, WeatherQuery) ────────
class MathQuery(BaseModel):
    operation: str = Field(
        ..., 
        description="연산 종류 ('add', 'subtract', 'multiply', 'divide', 'abs')"
    )
    num1: float = Field(..., description="첫 번째 숫자")
    num2: float | None = Field(
        default=None, 
        description="두 번째 숫자 (abs 연산 시 생략 가능)"
    )

    def calculate(self) -> str:
        """연산 수행 메서드"""
        op = self.operation.lower().strip()
        n1 = self.num1
        n2 = self.num2

        if op in ["add", "더하기", "+"]:
            return f"{n1} + {n2} = {n1 + (n2 if n2 is not None else 0)}"
        elif op in ["subtract", "빼기", "-"]:
            return f"{n1} - {n2} = {n1 - (n2 if n2 is not None else 0)}"
        elif op in ["multiply", "곱하기", "*"]:
            return f"{n1} * {n2} = {n1 * (n2 if n2 is not None else 1)}"
        elif op in ["divide", "나누기", "/"]:
            if n2 == 0:
                return "❌ 오류: 0으로 나눌 수 없습니다."
            return f"{n1} / {n2} = {n1 / (n2 if n2 is not None else 1)}"
        elif op in ["abs", "절댓값"]:
            if n2 is not None:
                # 2 - 17 과 같이 두 수의 차의 절댓값 계산 지원
                diff = n1 - n2
                return f"|{n1} - {n2}| = |{diff}| = {abs(diff)}"
            return f"|{n1}| = {abs(n1)}"
        else:
            return f"❌ 지원하지 않는 연산자입니다: {self.operation}"


class WeatherQuery(BaseModel):
    location: str = Field(..., description="도시 또는 지역명 (예: 서울, 도쿄, 뉴욕, 부산, 제주)")
    date: str = Field(default="오늘", description="조회 날짜 (예: 오늘, 내일, 2026-08-26)")
    unit: str = Field(default="celsius", description="온도 단위 ('celsius' 또는 'fahrenheit')")

    def get_info(self) -> str:
        """지역/날짜 날씨 정보 요약 반환 메서드"""
        weather_db = {
            "서울": {"temp": "24°C", "condition": "맑음 ☀️", "humidity": "50%"},
            "도쿄": {"temp": "26°C", "condition": "흐림 ☁️", "humidity": "65%"},
            "뉴욕": {"temp": "18°C", "condition": "비 🌧️", "humidity": "80%"},
            "부산": {"temp": "25°C", "condition": "맑음 ☀️", "humidity": "58%"},
            "제주": {"temp": "27°C", "condition": "바람 💨", "humidity": "60%"},
        }
        info = weather_db.get(self.location, {"temp": "22°C", "condition": "쾌청 🌤️", "humidity": "55%"})
        return f"[{self.location} ({self.date})] 날씨: {info['condition']}, 기온: {info['temp']} (단위: {self.unit}), 습도: {info['humidity']}"


# ── 3. LangChain 도구(@tool) 정의 ───────────────────────────
@tool(args_schema=MathQuery)
def math_tool(operation: str, num1: float, num2: float | None = None) -> str:
    """ 수학 사칙연산 및 절댓값(abs) 계산을 수행하는 도구 """
    query = MathQuery(operation=operation, num1=num1, num2=num2)
    return query.calculate()


@tool(args_schema=WeatherQuery)
def weather_tool(location: str, date: str = "오늘", unit: str = "celsius") -> str:
    """ 지정된 지역과 날짜의 날씨 및 기온 정보를 조회하는 도구 """
    query = WeatherQuery(location=location, date=date, unit=unit)
    return query.get_info()


tool_dict = {
    "math_tool": math_tool,
    "weather_tool": weather_tool
}
tools = [math_tool, weather_tool]


# ── 4. 사이드바 (Sidebar) UI ────────────────────────────────
with st.sidebar:
    st.title("⚙️ 시스템 설정")
    st.markdown("---")

    # API 키 설정
    default_api_key = os.getenv("OPENROUTER_API_KEY", "")
    api_key_input = st.text_input(
        "🔑 OpenRouter API Key",
        value=default_api_key,
        type="password",
        help=".env 파일에서 로드되거나 직접 입력할 수 있습니다."
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

    # 창의성(Temperature)
    temperature = st.slider("🌡️ Temperature (창의성)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    st.markdown("---")
    st.subheader("🛠️ 등록된 도구")
    st.info(
        "**1. math_tool (MathQuery)**\n"
        "- 사칙연산(+, -, *, /) 및 절댓값(abs)\n\n"
        "**2. weather_tool (WeatherQuery)**\n"
        "- 도시/지역별 날씨 및 기온 정보"
    )

    st.metric("📊 총 질문 횟수", f"{st.session_state.query_count} 회")

    if st.button("🗑️ 대화 세션 초기화 (Clear Session)", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.rerun()


# ── 5. 메인 화면 UI 및 LCEL 파이프라인 ───────────────────────
st.title("📘 AI 에이전트 도구 시스템 (`test2.py`)")
st.caption("Pydantic 스키마(MathQuery.calculate, WeatherQuery.get_info) + LCEL + OpenRouter 연동")

# 이전 대화 히스토리 화면 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "tool_info" in msg and msg["tool_info"]:
            with st.expander("🔍 도구 실행 세부 정보", expanded=False):
                st.code(msg["tool_info"], language="text")

# 빠른 질문 버튼
col1, col2, col3 = st.columns(3)
quick_query = None
if col1.button("🧮 abs(2 - 17) 계산해줘"):
    quick_query = "abs(2 - 17) 계산해줘"
if col2.button("🌤️ 서울 오늘 날씨 알려줘"):
    quick_query = "서울의 오늘 날씨는 어때?"
if col3.button("✨ 도쿄 날씨 및 35 * 12 계산"):
    quick_query = "도쿄의 날씨를 알려주고, 35 곱하기 12의 값도 알려줘."

# 사용자 입력 받기
user_input = st.chat_input("질문을 입력하세요 (예: abs(5-20) 계산, 제주도 날씨)...")
query_to_run = user_input or quick_query

if query_to_run:
    if not api_key_input:
        st.error("⚠️ OpenRouter API Key를 사이드바에 입력하거나 .env 파일에 등록해 주세요.")
        st.stop()

    # 1. 사용자 질문을 세션 상태에 저장 및 출력
    st.session_state.messages.append({"role": "user", "content": query_to_run})
    st.session_state.query_count += 1
    with st.chat_message("user"):
        st.markdown(query_to_run)

    # 2. 대화 기록을 LangChain 메시지 객체로 변환 (History)
    chat_history = []
    for m in st.session_state.messages[:-1]:  # 현재 질문 제외한 이전 기록
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            chat_history.append(AIMessage(content=m["content"]))

    # 3. 모델 및 도구 바인딩 (OpenRouter)
    model = ChatOpenAI(
        model=selected_model,
        api_key=api_key_input,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature
    )
    model_with_tools = model.bind_tools(tools)

    tool_logs = []

    # 4. 도구 실행 함수 (execute_tool_calls)
    def execute_tool_calls(ai_message):
        """AI의 도구 호출 요청을 감지하여 실행 후 최종 답변 생성"""
        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            tool_results = []
            for tool_call in ai_message.tool_calls:
                t_name = tool_call["name"]
                t_args = tool_call["args"]
                tool_logs.append(f"🛠️ 도구: {t_name}\n📌 인자: {t_args}")

                selected_tool = tool_dict.get(t_name)
                if selected_tool:
                    tool_msg = selected_tool.invoke(tool_call)
                    tool_results.append(tool_msg)
                    tool_logs.append(f"📊 실행 결과: {tool_msg.content}\n" + "-"*40)

            # 도구 실행 결과를 바탕으로 2차 답변 생성
            final_messages = [
                SystemMessage("당신은 전달받은 도구 실행 결과를 바탕으로 사용자에게 친절하고 명확하게 한국어로 답변하는 AI 어시스턴트입니다."),
                *chat_history,
                HumanMessage(content=query_to_run),
                ai_message,
                *tool_results
            ]
            return model.invoke(final_messages)
        return ai_message

    # 5. LCEL 프롬프트 템플릿 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 사용자의 질문에 답변하기 위해 수학(math_tool) 및 날씨(weather_tool) 도구를 적극적으로 활용하는 유능한 AI 어시스턴트입니다."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    # 6. LCEL 체인 생성 (prompt | model_with_tools | execute_tool_calls | StrOutputParser)
    lcel_chain = (
        prompt
        | model_with_tools
        | RunnableLambda(execute_tool_calls)
        | StrOutputParser()
    )

    # 7. 실행 및 결과 출력
    with st.chat_message("assistant"):
        with st.spinner("AI 분석 및 도구 실행 중..."):
            try:
                final_answer = lcel_chain.invoke({
                    "input": query_to_run,
                    "chat_history": chat_history
                })
                st.markdown(final_answer)

                tool_info_str = "\n".join(tool_logs) if tool_logs else ""
                if tool_info_str:
                    with st.expander("🔍 도구 실행 세부 정보", expanded=False):
                        st.code(tool_info_str, language="text")

                # 세션 상태에 AI 답변 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "tool_info": tool_info_str
                })
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
