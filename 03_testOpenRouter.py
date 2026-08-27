import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. .env 파일에서 환경변수 로드
load_dotenv()

# 2. OpenRouter API Key 가져오기
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

# 3. OpenRouter를 이용한 ChatOpenAI 모델 생성
# base_url을 https://openrouter.ai/api/v1 로 지정하면 OpenRouter의 모든 모델을 사용할 수 있습니다.
model = ChatOpenAI(
    model="openai/gpt-4o-mini",                     # OpenRouter에서 지원하는 모델 ID
    # model="google/gemini-2.5-flash",              # 예: Gemini 모델
    # model="deepseek/deepseek-chat",               # 예: DeepSeek 모델
    # model="meta-llama/llama-3.3-70b-instruct",    # 예: LLaMA 모델
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7
)

# 4. 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절하고 전문적인 인공지능 AI 선생님입니다. 사용자의 질문에 한국어로 친절하고 명확하게 답해주세요."),
    ("user", "{ask}에 대해서 설명해줘")
])

# 5. LCEL (LangChain Expression Language) 체인 구성 및 실행
print("=== OpenRouter 모델 호출 테스트 ===")
chain = prompt | model | StrOutputParser()

response = chain.invoke({"ask": "제주도"})
print(response)
