# test.py 구조도 및 Streamlit + LCEL 가이드

`test.py`는 **Streamlit 웹 UI**, **`st.session_state` 세션 관리**, **Pydantic 스키마(`MathQuery`, `WeatherQuery`)**, 그리고 **LangChain LCEL 파이프라인**을 결합한 지능형 AI 도구 에이전트 시스템입니다.

---

## 1. 전체 시스템 아키텍처 (Mermaid Flowchart)

```mermaid
flowchart TD
    User(["👤 사용자 브라우저"]) --> StreamlitApp["🌐 Streamlit 웹 애플리케이션 (test.py)"]
    
    subgraph UI_Layer ["🖥️ UI 및 세션 관리 계층"]
        Sidebar["⚙️ 사이드바 (st.sidebar)<br/>- 모델 및 API 키 설정<br/>- 세션 통계 및 기록 초기화"]
        MainChat["💬 채팅 인터페이스<br/>- st.session_state (대화 기록 보존)<br/>- st.chat_input / 빠른 질문 버튼"]
    end

    StreamlitApp --> UI_Layer

    subgraph LCEL_Engine ["⚡ LangChain LCEL 파이프라인"]
        Prompt["1. ChatPromptTemplate<br/>(사용자 질문 패키징)"]
        LLM["2. OpenRouter ChatOpenAI<br/>(도구 바인딩 llm_with_tools)"]
        ToolRouter{"도구 호출 필요 여부<br/>(tool_calls)"}
        
        subgraph ToolExecution ["🛠️ Pydantic 도구 실행"]
            MathTool["calculate_math<br/>(스키마: MathQuery)"]
            WeatherTool["get_current_weather<br/>(스키마: WeatherQuery)"]
        end
        
        Summarizer["3. 2차 요약 및 답변 완성<br/>(RunnableLambda)"]
        Parser["4. 문자열 파서<br/>(StrOutputParser)"]

        Prompt --> LLM --> ToolRouter
        ToolRouter -- "도구 필요" --> ToolExecution --> Summarizer --> Parser
        ToolRouter -- "일반 대화" --> Parser
    end

    MainChat --> LCEL_Engine
    Parser --> Display["✨ 최종 AI 답변 및 세부 정보 출력<br/>(st.chat_message & st.expander)"]
```

---

## 2. 대화 및 도구 호출 시퀀스 (Mermaid Sequence Diagram)

```mermaid
sequenceDiagram
    autonumber
    actor User as 사용자
    participant App as Streamlit UI (test.py)
    participant State as st.session_state
    participant LCEL as LCEL 체인 (Prompt | LLM)
    participant Tool as Pydantic 도구 (Math/Weather)

    User->>App: 질문 입력 ("서울 날씨는?")
    App->>State: 사용자 메시지 및 질의 횟수 저장
    App->>LCEL: chain.invoke({question: "서울 날씨는?"})
    
    Note over LCEL: 1차 LLM 호출: get_current_weather 도구 선택
    LCEL->>Tool: get_current_weather(location="서울") 실행
    Tool-->>LCEL: 결과 반환 ("[서울] 날씨: 맑음, 기온: 24°C")
    
    Note over LCEL: 2차 LLM 호출: 실행 결과 바탕으로 자연어 문장 생성
    LCEL-->>App: 최종 답변 텍스트 반환
    
    App->>State: 어시스턴트 답변 및 도구 로그 저장
    App-->>User: 화면에 AI 답변 및 도구 세부 정보(Expander) 출력
```

---

## 3. 핵심 구성 요소 상세 설명

| 계층 (Layer) | 구성 요소 | 역할 및 기능 |
| :--- | :--- | :--- |
| **세션 관리** | `st.session_state` | 새로고침 후에도 대화 내역(`messages`), 질의 횟수(`query_count`), 도구 사용 통계(`tool_usage_stats`)를 안전하게 유지 |
| **설정 UI** | `st.sidebar` | OpenRouter 모델 변경, API Key 입력, Temperature 조절, 대화 기록 다운로드 및 초기화 |
| **입력 스키마** | `MathQuery`, `WeatherQuery` | Pydantic `BaseModel`을 상속받아 AI가 도구를 호출할 때 필요한 파라미터 규격을 검증 |
| **도구 정의** | `@tool(args_schema=...)` | 파이썬 함수에 스키마를 바인딩하여 LLM이 사용할 수 있는 전용 툴로 변환 |
| **파이프라인** | **LCEL Chain** | `Prompt | LLM | RunnableLambda | Parser` 파이프 연산자로 데이터 흐름을 한 줄로 연결 |

---

## 4. LCEL 파이프라인 코드 요약

```python
# 파이프(|)로 연결된 직관적인 데이터 파이프라인
chain = (
    prompt                                     # 1. 프롬프트 생성
    | llm_with_tools                           # 2. 도구가 바인딩된 LLM 모델 호출
    | RunnableLambda(execute_tools_and_summarize) # 3. 도구 실행 및 최종 답변 합성
    | StrOutputParser()                        # 4. 순수 문자열 추출
)
```
