# -*- coding: utf-8 -*-
"""
딸깍콘(DdalkakCon) - Streamlit 앱 (프론트엔드 + 코드 통합)
담당: 프론트엔드 및 코드 통합

핵심 함수는 emoticon_tools.py 에 있고, 이 파일은
  - 화면 구성(사이드바 / 채팅 / CSS)
  - 업로드 이미지를 tool 쪽으로 전달
  - LangChain tool calling 루프 통합
을 담당한다.
"""

import base64
import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

import emoticon_tools as et
from emoticon_tools import TOOLS, TOOL_DICT

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(TOOLS)


# ----------------------------------------------------- AI 응답 처리

def get_ai_response(messages):
    """LLM 응답을 스트리밍하고, tool 호출 요청이 있으면 실행한 뒤 다시 호출한다."""
    response = llm_with_tools.stream(messages)

    # 스트리밍된 AIMessageChunk들을 하나의 완성된 메시지로 합치기 위한 변수
    gathered = None

    for chunk in response:
        # 화면에는 텍스트만 흘려보낸다 (chunk 객체를 그대로 주면 repr이 찍힌다)
        if chunk.content:
            yield chunk.content if isinstance(chunk.content, str) else str(chunk.content)

        gathered = chunk if gathered is None else gathered + chunk

    if gathered is None:
        return

    # 완성된 메시지에 Tool Call이 있다면 실제 Tool을 실행
    if gathered.tool_calls:
        st.session_state.messages.append(gathered)

        for tool_call in gathered.tool_calls:
            selected_tool = TOOL_DICT[tool_call["name"]]

            with st.spinner("🎨 이모티콘 프레임을 생성하고 있어요... (30~60초)"):
                tool_msg = selected_tool.invoke(tool_call)

            st.session_state.messages.append(tool_msg)

        # ToolMessage까지 추가된 새 대화 기록으로 모델을 다시 호출해 최종 답변 생성
        for piece in get_ai_response(st.session_state.messages):
            yield piece


# ----------------------------------------------------- 화면 기본 설정

st.set_page_config(
    page_title="💬 딸깍콘",
    page_icon="💬",   # 로컬 파일 경로를 쓰면 세션이 바뀔 때 깨지므로 이모지로 둔다
    layout="wide",
)

st.markdown(
    """
<style>
/* 전체 앱의 배경색을 살짝 밝은 회색/아이보리 톤으로 */
.stApp { background-color: #F8F9FA; }

/* 버튼 모양과 색상 */
div.stButton > button, div.stDownloadButton > button {
    background-color: #4F46E5 !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 8px 16px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    background-color: #3730A3 !important;
    transform: translateY(-2px);
    box-shadow: 0px 4px 12px rgba(79, 70, 229, 0.3) !important;
}

/* 카드 박스(st.metric) */
div[data-testid="stMetric"] {
    background-color: white !important;
    padding: 16px !important;
    border-radius: 16px !important;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.05) !important;
    border: 1px solid #E5E7EB !important;
}

/* 커스텀 태그 */
.highlight-title { color: #1E293B; font-size: 2.2rem; font-weight: 800; margin-bottom: 0px; }
.custom-badge {
    background-color: #E0E7FF; color: #3730A3;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.85rem; font-weight: 600;
    display: inline-block; margin-bottom: 15px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------- 사이드바

with st.sidebar:
    st.header("🤔 이모티콘으로 만들 이미지")
    st.divider()

    uploaded_file = st.file_uploader(
        "캐릭터 이미지 업로드 (선택)", type=["png", "jpg", "jpeg", "webp"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="업로드된 이미지", use_container_width=True)
        # 업로드 이미지를 tool이 꺼내 쓸 수 있도록 등록한다.
        # Streamlit은 매 상호작용마다 스크립트를 위에서부터 다시 실행하므로
        # 여기서 등록해두면 아래 채팅 처리 시점에 항상 최신 이미지가 들어가 있다.
        et.set_reference_image(uploaded_file.getvalue(), uploaded_file.type)
    else:
        et.set_reference_image(None)
        st.caption("이미지를 안 올려도 설명만으로 캐릭터를 새로 그려줘요!")

    st.divider()
    st.header("🎁 완성된 이모티콘")

    if os.path.exists(et.GIF_PATH):
        st.image(et.GIF_PATH, caption=f"{et.CANVAS}x{et.CANVAS} 무한루프 이모티콘",
                 use_container_width=True)

        info = et.last_result
        if info:
            c1, c2 = st.columns(2)
            c1.metric("총 프레임", f"{info.get('total_frames', '-')}장")
            c2.metric("재생 간격", f"{info.get('duration_ms', '-')}ms")
            st.caption(
                f"원본 {info.get('unique_frames', '-')}장 → "
                f"{info.get('loop_mode', '-')} 루프 / "
                f"GIF {info.get('gif_size', 0) / 1024:.0f}KB"
            )

        with open(et.GIF_PATH, "rb") as file:
            st.download_button(
                label="📥 제안용 다운로드 (.gif)",
                data=file.read(),
                file_name="ddalkakcon_emoticon.gif",
                mime="image/gif",
                use_container_width=True,
            )

        if os.path.exists(et.WEBP_PATH):
            with open(et.WEBP_PATH, "rb") as file:
                st.download_button(
                    label="📥 승인 후 제출용 (.webp)",
                    data=file.read(),
                    file_name="ddalkakcon_emoticon.webp",
                    mime="image/webp",
                    use_container_width=True,
                )

        frame_paths = [p for p in et.last_result.get("frame_paths", []) if os.path.exists(p)]
        if frame_paths:
            with st.expander(f"🖼️ 생성된 원본 프레임 {len(frame_paths)}장 보기"):
                st.image(frame_paths, width=110)
    else:
        st.info("💡 채팅창에서 이모티콘 생성을 요청하면 이곳에 미리보기와 다운로드 버튼이 나타납니다!")


# ----------------------------------------------------- 본문 헤더

st.markdown('<div class="custom-badge">🤣 딸깍!으로 만드는 이모티콘</div>',
            unsafe_allow_html=True)
st.markdown('<h1 class="highlight-title">💬 딸깍콘</h1>', unsafe_allow_html=True)
st.caption(
    "캐릭터 이미지와 설명을 주면 → 기획 → 프레임 생성 → 투명배경 정리 → "
    f"{et.CANVAS}×{et.CANVAS} 무한루프 이모티콘 파일까지 한 번에 만들어 드려요."
)


# ----------------------------------------------------- 시스템 프롬프트

SYSTEM_PROMPT = """[Role & Goal]
당신은 사용자의 요청을 바탕으로 세심하고 매력적인 '움직이는 이모티콘(Animated Emoticon)'을 기획하고, create_animated_emoticon 도구로 실제 파일까지 만들어 주는 친절한 AI 전문 디자이너입니다.

[Persona & Tone]
- 항상 밝고 다정하며 친근한 어조로 대화합니다. (예: "~해요!", "~해드릴게요 😊")
- 사용자의 아이디어를 적극적으로 칭찬하고 긍정적인 반응을 보입니다.
- 전문 용어보다는 누구나 이해하기 쉽고 따뜻한 표현을 씁니다.

[Core Instructions]
1. 사용자 프롬프트 최우선 반영
   - 캐릭터 외형, 동작, 감정, 색상, 분위기, 텍스트 등 사용자가 준 조건을 하나도 놓치지 않습니다.
   - 요청이 구체적이지 않으면 의도를 해치지 않는 선에서 매력적인 연출을 더해줍니다.

2. 먼저 기획안을 제시합니다
   - 캐릭터 디자인(외형/색상/분위기), 주요 감정과 표정, 프레임별 자세, 시각 효과를 설명합니다.
   - 프레임은 '시작 자세 → 핵심 동작 → 돌아오기' 흐름으로 6~8단계를 묘사합니다.

3. 도구 호출 규칙 (중요)
   - 사용자가 "만들어줘", "생성해줘", "좋아 이걸로 해줘" 등 제작을 요청하면
     반드시 create_animated_emoticon 도구를 호출합니다.
   - 사용자가 기획안만 보고 싶어하면 도구를 호출하지 말고 대화로만 답합니다.
   - 도구에 넘기는 character_description 과 frame_steps 는 반드시 '영문'으로 작성합니다.
     이미지 생성 AI가 영문 프롬프트에서 품질이 더 좋기 때문입니다.
   - frame_steps 의 각 문장은 '한 장의 정지 그림'을 묘사합니다. 움직임을 설명하지 않고
     그 순간의 자세와 표정만 적습니다. 배경이나 화면 구도는 언급하지 않습니다
     (도구가 투명배경과 규격을 알아서 처리합니다).
   - 반복 재생되므로 첫 번째와 마지막 프레임은 같은 기본 자세로 둡니다.
     마지막 자세가 썸네일이 되니 캐릭터를 가장 잘 보여주는 자세로 합니다.
   - 효과(하트·반짝임·눈물·땀·김·음표)와 한글 글자는 frame_steps 에 쓰지 말고
     effect 와 overlay_text 인자로 넘깁니다. 그래야 정확한 위치에 또렷하게 들어갑니다.

4. 도구 실행 후
   - 결과 메시지를 바탕으로 무엇이 만들어졌는지 친절하게 요약합니다.
   - 왼쪽 사이드바에서 미리보기와 다운로드가 가능하다고 안내합니다.
   - 수정하고 싶은 점이 있으면 말해달라고 덧붙입니다.
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [SystemMessage(content=SYSTEM_PROMPT)]


# ----------------------------------------------------- 대화 기록 출력

def display_user_content(content):
    """HumanMessage는 텍스트만, 또는 텍스트+이미지 리스트일 수 있다."""
    if isinstance(content, str):
        st.write(content)
        return
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                st.write(item.get("text", ""))
            elif item.get("type") == "image_url":
                img_url = item.get("image_url", {}).get("url", "")
                if img_url:
                    st.image(img_url, caption="첨부한 이미지", width=200)


for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue

    elif isinstance(msg, AIMessage):
        if msg.content:
            st.chat_message("assistant").write(msg.content)

    elif isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            display_user_content(msg.content)

    elif isinstance(msg, ToolMessage):
        with st.chat_message("assistant", avatar="🛠️"):
            with st.expander("이모티콘 생성 도구 실행 결과"):
                st.code(msg.content, language=None)


# ----------------------------------------------------- 사용자 입력 처리

if prompt := st.chat_input("예: 이 캐릭터가 신나서 점프하는 이모티콘 만들어줘!"):
    if uploaded_file is not None:
        base64_image = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
        user_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{uploaded_file.type};base64,{base64_image}"},
            },
        ]
        user_message = HumanMessage(content=user_content)
    else:
        user_message = HumanMessage(content=prompt)

    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append(user_message)

    with st.chat_message("assistant"):
        result = st.write_stream(get_ai_response(st.session_state["messages"]))

    if result:
        st.session_state["messages"].append(AIMessage(content=result))

    # 사이드바가 새로 만들어진 이모티콘을 바로 보여줄 수 있도록 다시 그린다
    st.rerun()
