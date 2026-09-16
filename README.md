# PDF RAG Lab

PDF 문서를 페이지 단위로 읽고, 질문과 관련된 청크를 검색하여 근거 페이지와 함께 답변하는 RAG 서비스를 단계별로 구현하는 학습 프로젝트입니다.

완성 코드를 빠르게 만드는 것보다 RAG를 구성하는 파싱, 청킹, 임베딩, 검색, 평가, 재정렬, 생성이 왜 필요한지 직접 이해하고 구현하는 데 초점을 둡니다. LangChain이나 LlamaIndex 없이 핵심 계산과 데이터 흐름을 작은 단위로 만들고 테스트합니다.

## 현재 구현

- PyMuPDF를 이용한 PDF 검증, 페이지별 텍스트 추출, 공백과 빈 줄 정규화
- 페이지 정보를 보존하는 문자 수 기반 고정 크기·overlap 청킹
- 공통 단어 비율로 점수를 계산하고 정렬하는 키워드 검색 baseline과 `top_k` 검증
- 내적과 벡터 norm으로 직접 구현한 코사인 유사도
- 가짜 벡터를 이용해 검증한 dense retrieval
- OpenAI `text-embedding-3-small`을 이용한 단일·배치 텍스트 임베딩
- 원본 청크를 변경하지 않고 임베딩을 복사본에 결합하는 `embed_chunks`
- Hit@k와 RR 계산 및 여러 질문의 평균인 MRR 평가 기반
- 기존 후보의 검색 점수를 보존하면서 별도의 가짜 rerank 점수로 재정렬하는 `rerank_results`
- PDF 업로드 후 페이지와 청크를 반환하는 `POST /papers` API

아직 검색 API, 답변 생성, 근거 페이지를 포함한 최종 답변, 실제 AI Reranker 모델은 구현하지 않았습니다. 현재 reranker는 외부에서 받은 가짜 점수로 정렬 로직만 검증합니다.

## 디렉터리 구조

```text
pdf-rag-lab/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── parser.py
│   ├── chunker.py
│   ├── retrieval.py
│   ├── embedder.py
│   ├── evaluation.py
│   └── reranker.py
├── scripts/
│   ├── __init__.py
│   └── experiment.py
├── tests/
│   ├── test_api.py
│   ├── test_parser.py
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   ├── test_embedder.py
│   ├── test_evaluation.py
│   └── test_reranker.py
├── samples/
├── .gitignore
├── requirements.txt
└── README.md
```

`app/`에는 서비스와 RAG 구성 요소가, `scripts/`에는 실제 문서 실험 코드가, `tests/`에는 API 및 단위 테스트가 있습니다. `samples/`의 PDF와 `cache/`의 임베딩 결과는 로컬 실험 자료이므로 Git에 포함하지 않습니다.

## RAG 파이프라인과 진행 위치

```text
PDF 업로드 → 페이지별 파싱 → 정규화 → 페이지 추적 청킹
  → 임베딩 → dense retrieval → 평가 → reranking → 답변 생성·근거 제시
       완료          완료         완료    로직만 완료       다음 단계
```

키워드 검색은 dense retrieval과 비교하기 위한 baseline입니다. 현재 주된 실험 위치는 dense retrieval 평가 이후와 실제 reranker 연결 이전입니다.

## 설치와 실행

Python 가상환경을 만들고 프로젝트 루트에서 의존성을 설치합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

개발 서버를 실행합니다.

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

로컬 Swagger UI는 `http://127.0.0.1:8000/docs`에서 확인할 수 있습니다. GitHub Codespaces에서는 포트 8000의 전달된 주소에 `/docs`를 붙입니다.

## OpenAI API 키

임베딩 실험에는 `OPENAI_API_KEY`가 필요합니다. GitHub Codespaces에서는 저장소의 **Settings → Secrets and variables → Codespaces**에서 이름을 `OPENAI_API_KEY`로 등록한 뒤 Codespace를 다시 시작합니다. 로컬 셸에서는 현재 세션의 환경변수로 설정할 수 있습니다.

```bash
export OPENAI_API_KEY="your-api-key"
```

API 키를 코드에 직접 쓰거나 Git에 커밋하지 마세요. 로컬에서 `.env`를 사용하더라도 해당 파일을 커밋하지 마세요. 이 프로젝트는 `.env`를 자동으로 읽지 않으므로 필요하면 셸에서 환경변수를 로드해야 합니다.

## 샘플 PDF와 실험

실험 문서는 RAG 원 논문인 [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401)입니다. PDF 파일은 저장소에 커밋하지 않으며 사용자가 직접 내려받습니다.

```bash
mkdir -p samples
curl -L https://arxiv.org/pdf/2005.11401 -o samples/rag_paper.pdf
```

API 호출 비용이 발생할 수 있는 실험은 명시적으로 다음 명령을 실행할 때만 수행합니다.

```bash
python -m scripts.experiment
```

현재 실험 코드는 이 19페이지 PDF를 기본 `chunk_size=1000`, `overlap=200`으로 처리하여 92개 청크를 만들고, 청크 임베딩을 `cache/rag_paper_embeddings.json`에 저장합니다. 캐시가 있으면 청크 임베딩은 다시 요청하지 않지만 평가 질문의 임베딩은 실행할 때마다 API로 생성합니다.

이 캐시는 설정을 검증하는 fingerprint가 없는 실험용 MVP입니다. 다음 항목이 바뀌면 오래된 값이 될 수 있으므로 `cache/rag_paper_embeddings.json`을 삭제하고 다시 생성해야 합니다.

- PDF 내용
- `chunk_size`
- `overlap`
- 임베딩 모델

캐시 JSON과 내려받은 PDF는 모두 `.gitignore`에서 제외합니다.

## Retrieval 평가

평가는 질문마다 정답으로 지정한 청크 인덱스가 검색 순위 어디에 나타나는지 측정합니다.

- **Hit@k**: 상위 k개 결과 안에 정답 청크가 하나라도 있으면 1, 없으면 0
- **RR (Reciprocal Rank)**: 첫 정답 청크의 순위가 r일 때 `1/r`; 정답이 없으면 0
- **MRR (Mean Reciprocal Rank)**: 여러 평가 질문에서 계산한 RR의 평균

현재 실험 스크립트에는 RAG 원 논문을 대상으로 한 평가 질문 3개와 각 질문의 정답 청크가 정의되어 있습니다. 기록된 dense retrieval baseline 결과는 다음과 같습니다.

- 평가 질문: 3개
- Mean Hit@3: `0.6667`
- MRR: `0.7083`

질문의 뜻이 비슷해도 표현에 따라 정답 청크 순위가 1위에서 8위까지 달라지는 사례를 관찰했습니다. 이는 소수 질문의 점수만으로 검색 품질을 일반화할 수 없고, 질문 표현을 다양하게 늘려 평가해야 함을 보여 줍니다.

## 테스트

테스트는 실제 OpenAI API를 호출하지 않습니다. 임베딩 결합 테스트는 가짜 임베딩 함수를 주입하고, dense retrieval 테스트는 고정 벡터를 사용합니다.

```bash
python -m pytest -v
```

테스트 범위에는 PDF 오류 처리와 텍스트 추출, 정규화, overlap 청킹과 페이지 추적, 업로드 API, 키워드 및 dense 검색, 입력 검증, 임베딩 결합의 불변성, Hit@k와 RR, 가짜 점수 reranking이 포함됩니다.

## 학습 기록

### Day 3: Dense Retrieval 평가와 프로젝트 구조 정리 (2026.09.13)

#### 구현하고 정리한 내용

- 키워드 검색 baseline, 점수 정렬, `top_k`, 입력 검증 상태 확인
- 코사인 유사도와 가짜 벡터 기반 dense retrieval 테스트 확인
- OpenAI `text-embedding-3-small` 배치 임베딩과 `embed_chunks` 확인
- Hit@k, RR, MRR 평가 코드와 평가 질문 3개 확인
- RAG 원 논문 19페이지를 92개 청크로 처리한 실험과 임베딩 JSON 캐시 문서화
- 가짜 rerank 점수로 후보를 재정렬하는 로직 확인
- 최상위 Python 모듈을 `app/` 패키지로, 실험 코드를 `scripts/` 패키지로 이동
- 애플리케이션, 테스트, 실험 스크립트의 import와 실행 명령 수정
- 직접 의존성을 `requirements.txt`에 기록
- 캐시, 샘플 PDF, 가상환경, API 키 파일을 Git에서 제외
- 실제 OpenAI API 호출 없이 전체 테스트 33개 통과

#### 배운 내용

Python 파일을 디렉터리로 옮길 때는 파일 위치만 바꾸는 것으로 끝나지 않습니다. 패키지를 나타내는 `__init__.py`를 추가하고 애플리케이션, 테스트, 스크립트, uvicorn 명령에서 같은 모듈 경로를 사용해야 합니다.

```bash
python -m uvicorn app.main:app --reload
python -m scripts.experiment
```

임베딩 캐시는 API 호출 비용과 시간을 줄이지만 입력과 설정이 달라졌는지 스스로 판단하지 못합니다. PDF 내용, `chunk_size`, `overlap`, 임베딩 모델이 바뀌면 기존 캐시를 삭제하고 다시 생성해야 합니다. 이후에는 이 값들로 fingerprint를 만들어 자동으로 무효화할 필요가 있습니다.

검색 평가는 단순히 정답 포함 여부만 보는 것보다 정답의 순위도 함께 확인해야 합니다. Hit@k는 상위 k개 안의 정답 존재 여부를, RR은 첫 정답의 순위를, MRR은 여러 질문의 평균 순위 품질을 보여 줍니다. 비슷한 뜻의 질문도 표현에 따라 정답 청크가 1위에서 8위로 달라질 수 있어 평가 질문을 늘리는 작업이 필요합니다.

현재 reranker는 실제 AI 모델을 호출하지 않습니다. 후보와 가짜 점수를 결합하여 점수순으로 정렬하는 데이터 흐름만 먼저 검증했습니다.

---

### Day 1: PDF 업로드와 파싱 (2026.09.11)

#### 구현한 내용

- FastAPI의 UploadFile을 이용한 PDF 업로드 API
- PyMuPDF를 이용한 페이지별 텍스트 추출
- PDF 파싱 로직과 API 로직 분리
- 잘못된 파일에 대한 예외 처리
- pytest를 이용한 파서 단위 테스트
- FastAPI TestClient를 이용한 API 테스트

#### 배운 내용

업로드된 파일은 bytes로 읽힙니다.

```python
pdf_bytes = await file.read()
```

`file.read()`는 파일 경로가 아니라 업로드된 파일의 실제 내용을 bytes 형태로 반환합니다.

PDF에 글자가 보인다고 항상 텍스트를 추출할 수 있는 것은 아닙니다. 일반적인 텍스트 PDF에는 실제 문자 데이터가 들어 있으므로 `page.get_text()`로 추출할 수 있습니다. 스캔한 문서는 글자가 이미지의 일부로 저장되어 있을 수 있습니다. 이 경우 화면에는 글자가 보여도 텍스트 추출 결과는 비어 있을 수 있으며 별도의 OCR 처리가 필요합니다.

`any()`로 전체 PDF에 텍스트가 있는지 확인할 수 있습니다.

```python
if not any(page_data["text"] for page_data in pages):
    raise PdfParsingError("PDF에서 텍스트를 찾을 수 없습니다.")
```

각 페이지의 텍스트 중 하나라도 비어 있지 않으면 `any()`는 `True`를 반환합니다. 모든 페이지가 비어 있으면 `PdfParsingError`를 발생시킵니다.

파서는 PDF 처리 과정에서 발생한 문제를 `PdfParsingError`로 표현합니다. API는 이 오류를 잡아 클라이언트가 이해할 수 있는 HTTP 400 응답으로 변환합니다.

```python
try:
    pages = extract_pages(pdf_bytes)
except PdfParsingError as error:
    raise HTTPException(
        status_code=400,
        detail=str(error),
    ) from error
```

`pytest.raises()`로 예상한 예외가 발생하는지 검증할 수 있습니다.

```python
with pytest.raises(PdfParsingError):
    extract_pages(b"")
```

#### 발생한 문제와 해결 과정

페이지 데이터에서 텍스트를 가져오는 코드를 `page_data[text]`로 작성하여 `KeyError`가 발생했습니다. 이 코드에서 `text`는 문자열 키가 아니라 이전 반복에서 저장된 페이지 본문을 가리키고 있었습니다. 딕셔너리의 실제 키인 `page_data["text"]`를 사용하도록 수정했습니다.

또한 서버 트레이스백은 아래쪽부터 살펴보면 실제 오류 종류와 오류가 발생한 파일 및 줄을 빠르게 찾을 수 있다는 점을 배웠습니다.

---

### Day 2: 페이지 추적 청킹 (2026.09.12)

#### 구현한 내용

- 문자 수 기반 `chunk_text()` 구현
- 청크 사이 overlap 적용
- 잘못된 청킹 설정에 대한 예외 처리
- 불필요한 마지막 중복 청크 방지
- 페이지 연결과 문자 범위를 기록하는 `combine_pages()` 구현
- 청크와 겹치는 페이지를 찾는 `find_page_numbers()` 구현
- 전체 청킹 과정을 조립하는 `chunk_pages()` 구현
- PDF 텍스트 공백과 빈 줄 전처리
- 청킹 결과를 `POST /papers` 응답에 연결
- 청킹과 API 통합 테스트 작성

#### 배운 내용

PDF 전체나 한 페이지 전체를 검색 단위로 사용하면 여러 주제가 하나의 검색 결과에 섞일 수 있습니다. 긴 텍스트를 작은 청크로 나누면 질문과 관련된 부분을 더 정밀하게 검색할 수 있습니다.

문장이나 중요한 정보가 청크 경계에서 잘릴 수 있습니다. 이전 청크의 마지막 일부를 다음 청크에도 포함하는 overlap으로 경계 주변의 문맥을 보존합니다.

잘못된 설정은 무한 반복을 만들 수 있습니다.

```python
step = chunk_size - overlap
```

`overlap`이 `chunk_size`보다 크거나 같으면 이동 거리가 0 이하가 됩니다. 시작 위치가 앞으로 이동하지 않아 반복문이 끝나지 않을 수 있으므로 해당 설정을 미리 거절합니다.

PDF의 페이지 경계는 의미 경계와 다를 수 있습니다. 문장이 다음 페이지로 이어질 수 있기 때문에 페이지별로 완전히 분리하여 청킹하면 문맥이 끊길 수 있습니다. 페이지들을 연결해 청킹하되, 각 페이지의 문자 범위를 별도로 기록하여 문맥과 출처 정보를 함께 보존합니다.

`page_numbers: [1, 2]`는 해당 청크가 1페이지와 2페이지의 내용을 포함한다는 뜻입니다. 실제 답변에 사용된 문장이 2페이지에만 있을 수도 있습니다. 현재 단계에서는 청크가 걸친 모든 페이지를 안전하게 반환하며, 답변 생성 단계에서 더 정밀한 인용 방식을 실험할 예정입니다.

PDF는 화면에 보이는 방식과 내부에 문자가 저장된 방식이 다를 수 있습니다. 실제 강의자료를 테스트하면서 다음 현상을 확인했습니다.

- 슬라이드의 줄바꿈이 다수의 `\n`으로 추출됨
- 특수 글꼴의 글머리표와 기호가 ``, `` 등의 문자로 추출됨
- 문자 수가 적은 슬라이드 여러 장이 하나의 청크에 포함됨

현재 전처리에서는 불필요한 공백과 빈 줄만 안전하게 제거합니다. 특수문자를 무조건 삭제하면 글머리표, 화살표, 수학 기호의 의미를 잃을 수 있으므로 추후 별도로 다룹니다. 일반 문서와 강의 슬라이드는 적절한 청킹 방식이 다를 수 있으므로 이후 검색 품질 평가에서 문서 유형, 청크 크기, 페이지 경계 처리 방식을 비교할 예정입니다.

#### 발생한 문제와 해결 과정

시스템에 설치된 pytest와 `.venv`의 Python이 서로 다른 환경을 사용하여 모듈을 찾지 못하는 문제가 발생했습니다. 현재 가상환경의 Python을 통해 테스트를 실행하도록 통일했습니다.

```bash
python -m pytest
```

`__pycache__`와 `.pyc` 파일은 Python이 자동으로 생성하며 실행할 때마다 변경됩니다. 해당 파일들을 `.gitignore`에 추가하고 Git의 추적 대상에서 제거했습니다.

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
```

당시 다음 학습 목표는 키워드 검색 baseline, 관련도 점수 정렬, Top-k 반환, 임베딩과 코사인 유사도 학습, 키워드 검색과 임베딩 검색 비교였습니다.

## 학습 원칙

1. 전체 기능을 한꺼번에 구현하지 않습니다.
2. 각 기능이 필요한 이유를 이해한 뒤 코드를 작성합니다.
3. 파싱, 청킹, 검색, 생성처럼 서로 다른 책임을 분리합니다.
4. 정상 동작과 실패 상황을 함께 테스트합니다.
5. 입력, 출력, 목적과 트레이드오프를 직접 설명할 수 있는 상태를 목표로 합니다.
6. 실험 결과와 실패 사례를 기록하고 다음 구현의 근거로 사용합니다.

## 다음 단계

1. 실제 AI Reranker를 연결합니다.
2. dense retrieval로 내부 후보 10개를 검색하고 최종 3개로 재정렬합니다.
3. 동일한 평가 질문으로 Hit@3와 MRR을 다시 측정합니다.
4. 평가 질문과 표현 변형을 확대합니다.
5. 캐시 fingerprint 또는 자동 무효화를 구현합니다.
6. 검색 결과를 바탕으로 답변을 생성하고 근거 페이지를 제공합니다.
