import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# 1. 웹페이지 기본 설정
st.set_page_config(page_title="Local RAG AI Tutor", layout="wide")
st.title("로컬 RAG 기반 PDF 문서 질의응답 AI 튜터")
st.write("인터넷 연결 없이 로컬 GPU 자원만 활용하여 데이터 유출을 방지하는 보안형 RAG(검색 증강 생성) 시스템입니다.")

# 2. 무료 임베딩 모델 최초 1회만 로드 (캐싱)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

embeddings = load_embeddings()

# 3. 업로드된 PDF 파일을 읽고 벡터 DB로 만드는 함수
@st.cache_resource(show_spinner=False)
def process_pdf_to_db(file_bytes, file_name):
    # Streamlit이 읽은 바이트 데이터를 임시 파일로 물리 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = text_splitter.split_documents(pages)
        vector_db = FAISS.from_documents(chunks, embeddings)
    finally:
        os.unlink(tmp_path) # 처리가 끝나면 임시 파일 삭제
        
    return vector_db

# 4. 사이드바에 동적 파일 업로더 배치
with st.sidebar:
    st.header("문서 등록")
    uploaded_file = st.file_uploader("분석할 PDF 전공 서적이나 문서를 업로드하세요.", type=["pdf"])
    
    if uploaded_file:
        st.success(f"{uploaded_file.name} 로드 완료!")
        vector_db = process_pdf_to_db(uploaded_file.getvalue(), uploaded_file.name)
        st.info("AI 조교가 문서 분석을 끝냈습니다. 대화를 시작하세요!")
    else:
        vector_db = None

# 5. 대화 기록 기억 바구니 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 리셋 버튼 (사이드바)
if st.sidebar.button("대화 기록 초기화"):
    st.session_state.messages = []
    st.rerun()

# 화면에 이전 대화 내용 유지하기
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 6. 사용자가 질문을 입력했을 때 작동
if query := st.chat_input("업로드한 문서 내용에 대해 질문해보세요!"):
    
    # PDF가 업로드되지 않았다면 경고 후 중단
    if vector_db is None:
        st.warning("먼저 왼쪽 사이드바에서 PDF 전공 서적을 업로드해 주세요!")
        st.stop()

    # 유저 질문 출력 및 저장
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # AI 답변 생성 프로세스
    with st.chat_message("assistant"):
        with st.spinner("교재 내용을 분석하고 문맥을 기억하는 중..."):
            
            # 관련 문서 검색 및 페이지 추출
            relevant_docs = vector_db.similarity_search(query, k=3)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # 사람이 읽기 편하게 1을 더해 실제 페이지 번호로 보정
            source_pages = sorted(list(set([doc.metadata.get('page', 0) + 1 for doc in relevant_docs])))

            # 최근 대화 맥락 요약해서 짚어주기 (메모리 기능)
            history_context = ""
            for msg in st.session_state.messages[-5:-1]: 
                history_context += f"{'학생' if msg['role']=='user' else 'AI튜터'}: {msg['content']}\n"

            # 로컬 LLM Llama3 연동
            llm = Ollama(model="llama3", temperature=0.2)
            
            prompt = f"""당신은 제공된 문서 내용을 바탕으로 학생을 가르치는 친절한 AI 조교입니다.
            반드시 아래 제공된 [교재 내용]만을 바탕으로 질문에 정확하고 풍부하게 답변하세요. 
            [이전 대화 기록]이 있다면 그 맥락을 고려하여 이어지는 대답을 작성하세요.
            모든 대답은 정중한 한국어(존댓말)로 논리정연하게 작성하고, 교재에 없는 사실을 지어내지 마세요.

            [이전 대화 기록]
            {history_context}

            [교재 내용]
            {context}

            학생의 질문: {query}
            AI 조교의 답변:"""

            # 최종 답변 출력
            response = llm.invoke(prompt)
            st.write(response)
            
            # 출처 표기 시각화
            page_tags = ", ".join([f"**{p}페이지**" for p in source_pages])
            st.markdown(f"근거 자료: {page_tags} 근처에서 내용을 발췌함.")
            
    # AI 답변 기록 저장
    st.session_state.messages.append({"role": "assistant", "content": response})