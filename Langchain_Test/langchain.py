#!/usr/bin/env python3
"""LangChain 문서 기반 챗봇 예제 스크립트

설치:
    pip install -r requirements.txt

사용 방법:
    python langchain.py

파일 경로 및 질문은 스크립트 내부 또는 커맨드라인 인자를 통해 수정하세요.
"""

import os
import time
from pathlib import Path
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OpenAI API 키 설정
os.environ["GOOGLE_API_KEY"] = "AIzaSyCW9zxkH8ei7jzCQgGlU2x0hseJmnnBTyc"


def load_pdf(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def load_text(txt_path: str, encoding: str = "utf-8") -> List[Document]:
    loader = TextLoader(txt_path, encoding=encoding)
    return loader.load()


def load_docx(docx_path: str) -> List[Document]:
    loader = UnstructuredWordDocumentLoader(docx_path)
    return loader.load()


def load_documents_from_data_dir(data_dir: Path) -> List[Document]:
    supported_ext = {".pdf", ".txt", ".docx"}
    docs: List[Document] = []

    if not data_dir.exists():
        print(f"[WARN] data 디렉토리가 없습니다: {data_dir}. 생성합니다.")
        data_dir.mkdir(parents=True, exist_ok=True)
        return docs

    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        if ext not in supported_ext:
            print(f"[WARN] 지원되지 않는 파일 형식입니다: {path.name}")
            continue

        try:
            if ext == ".pdf":
                loaded = load_pdf(str(path))
            elif ext == ".txt":
                for enc in ("utf-8", "cp949", "euc-kr"):
                    try:
                        loaded = load_text(str(path), encoding=enc)
                        break
                    except Exception:
                        loaded = []
            elif ext == ".docx":
                loaded = load_docx(str(path))
            else:
                loaded = []

            print(f"[INFO] 로드 완료: {path.name} ({len(loaded)} 문서)")
            docs.extend(loaded)
        except Exception as exc:
            print(f"[ERROR] {path.name} 로드 실패: {exc}")

    return docs


def split_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def build_faiss_index(documents: List[Document]) -> FAISS:
    """배치 처리 + 재시도로 rate limit 대응하며 FAISS 인덱스 생성"""
    from langchain_google_genai._common import GoogleGenerativeAIError

    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    batch_size = 10  # 한 번에 처리할 청크 수
    max_retries = 5

    all_embeddings = []
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"[INFO] 임베딩 중... ({i + 1}~{min(i + batch_size, len(texts))}/{len(texts)}개)")

        for attempt in range(max_retries):
            try:
                batch_embeddings = embeddings_model.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                time.sleep(1)  # 기본 1초 대기 (rate limit 방지)
                break
            except GoogleGenerativeAIError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_sec = 2 ** (attempt + 2)  # 4, 8, 16, 32, 64초
                    print(f"[WARN] Rate limit 초과. {wait_sec}초 후 재시도... (시도 {attempt + 1}/{max_retries})")
                    time.sleep(wait_sec)
                else:
                    raise
        else:
            raise RuntimeError(f"배치 {i}~{i+batch_size} 임베딩 실패: 최대 재시도 횟수 초과")

    print(f"[INFO] 전체 임베딩 완료: {len(all_embeddings)}개")
    return FAISS.from_embeddings(
        text_embeddings=list(zip(texts, all_embeddings)),
        embedding=embeddings_model,
        metadatas=metadatas,
    )


def build_rag_chain(retriever, model_name: str = "gemini-2.0-flash", temperature: float = 0.0):
    prompt = PromptTemplate.from_template(
        """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다.
검색된 다음 문맥(context)을 사용하여 질문(question)에 답하세요.
만약, 주어진 문맥(context)에서 답을 찾을 수 없다면, `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다`라고 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question:
{question}

#Context:
{context}

#Answer:"""
    )

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def build_rag_chain_with_sources(retriever, model_name: str = "gemini-2.0-flash", temperature: float = 0.0):
    prompt = PromptTemplate.from_template(
        """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다.
검색된 다음 문맥(context)을 사용하여 질문(question)에 답하세요.
만약, 주어진 문맥(context)에서 답을 찾을 수 없다면, `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다`라고 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.
반드시 출처도 제공해주세요.

#Question:
{question}

#Context:
{context}

#사용된 문서 유형:
{sources}

#Answer:"""
    )

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    def process_retrieval(question: str):
        docs = retriever.invoke(question)
        sources = ", ".join(sorted({doc.metadata.get("source", "Unknown") for doc in docs}))
        context_text = "\n".join(doc.page_content for doc in docs)
        return {"context": context_text, "question": question, "sources": sources}

    return (
        RunnableLambda(process_retrieval)
        | prompt
        | llm
        | StrOutputParser()
    )


def add_document_to_vectorstore(vectorstore: FAISS, text: str, splitter: RecursiveCharacterTextSplitter):
    new_doc = Document(page_content=text)
    split_docs = splitter.split_documents([new_doc])
    vectorstore.add_documents(split_docs)
    return len(split_docs)


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] LangChain 문서 기반 챗봇 스크립트 실행")
    print("[INFO] Google Gemini API 키가 설정되었습니다.")
    print(f"[INFO] data 폴더 경로: {data_dir}")

    loaded_docs = load_documents_from_data_dir(data_dir)
    if not loaded_docs:
        print("[WARN] data 폴더에 로드할 문서가 없습니다. PDF, TXT, DOCX 파일을 추가하세요.")
        print("[INFO] 스크립트 실행 종료")
        return

    split_docs = split_documents(loaded_docs, chunk_size=1000, chunk_overlap=100)
    print(f"[INFO] 총 문서 청크 생성 완료: {len(split_docs)}개")

    # FAISS 인덱스 저장/로드 (재실행 시 임베딩 재생성 방지)
    faiss_index_dir = base_dir / "faiss_index"
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    if faiss_index_dir.exists():
        print("[INFO] 저장된 FAISS 인덱스를 로드합니다...")
        vectorstore = FAISS.load_local(
            str(faiss_index_dir),
            embeddings=embeddings_model,
            allow_dangerous_deserialization=True,
        )
        print("[INFO] FAISS 인덱스 로드 완료")
    else:
        vectorstore = build_faiss_index(split_docs)
        vectorstore.save_local(str(faiss_index_dir))
        print(f"[INFO] FAISS 인덱스를 저장했습니다: {faiss_index_dir}")

    retriever = vectorstore.as_retriever()

    rag_chain = build_rag_chain(retriever)

    print("\n[질문 예시]: 금융기관에 대해서 분류해줘.")
    print("[질문 예시]: 주택 임대시 주의점은 무엇인가요?")

    # 대화형 QA 루프
    print("\n" + "=" * 50)
    print("[INFO] 직접 질문해보세요. 종료하려면 'q' 또는 '종료'를 입력하세요.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] 종료합니다.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit", "종료"):
            print("[INFO] 종료합니다.")
            break

        try:
            response = rag_chain.invoke(user_input)
            print(f"응답: {response}")
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                if "PerDay" in err or "per_day" in err.lower() or "daily" in err.lower():
                    print("[ERROR] 오늘의 무료 임베딩 할당량(1,000회)을 모두 사용했습니다.")
                    print("[INFO]  → 내일 다시 실행하면 저장된 인덱스를 사용해 바로 질문 가능합니다.")
                    print("[INFO]  → 또는 Google AI Studio에서 결제를 활성화하면 한도가 늘어납니다.")
                    print("[INFO]  → https://aistudio.google.com")
                    break
                else:
                    # 분당 rate limit인 경우 잠시 대기 후 재시도
                    import re
                    retry_match = re.search(r"retry[^\d]*(\d+)", err)
                    wait_sec = int(retry_match.group(1)) + 5 if retry_match else 60
                    print(f"[WARN] Rate limit 초과. {wait_sec}초 후 자동 재시도합니다...")
                    time.sleep(wait_sec)
                    try:
                        response = rag_chain.invoke(user_input)
                        print(f"응답: {response}")
                    except Exception as e2:
                        print(f"[ERROR] 재시도 실패: {e2}")
            else:
                print(f"[ERROR] 오류 발생: {e}")


if __name__ == "__main__":
    main()