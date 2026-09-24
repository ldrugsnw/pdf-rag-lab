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
- 기존 후보의 코사인 유사도 `score`를 보존하면서 별도의 `rerank_score`로 재정렬하는 `rerank_results`
- GPT-5-nano 기반 LLM Reranker, JSON Schema Structured Output, candidate 검증과 bounded retry
- Retriever Top 10 → Reranker Top 3 → GPT-5-nano 답변 생성 실험 파이프라인
- 페이지 정보를 포함한 context와 문서에 답이 없으면 답을 찾을 수 없다고 응답하도록 지시하는 생성 prompt
- PDF 업로드 후 페이지와 청크를 반환하는 `POST /papers` API

검색·답변 생성은 현재 실험 스크립트에서 실행하며, HTTP API는 `POST /papers` 업로드·파싱·청킹까지만 제공합니다. Generator는 context에 청크의 페이지 정보를 전달하지만, 최종 답변의 페이지 인용을 강제하거나 검증하는 기능은 아직 없습니다. Reranker 검증이 반복 실패하여 `RerankingError`가 발생하면 파이프라인은 Retriever Top 3으로 fallback합니다.

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
│   ├── reranker.py
│   └── generator.py
├── scripts/
│   ├── __init__.py
│   ├── experiment.py
│   ├── inspect_chunks.py
│   ├── real_reranker_experiment.py
│   ├── reranker_evaluation.py
│   ├── run_rag.py
│   └── test_grounding.py
├── tests/
│   ├── test_api.py
│   ├── test_parser.py
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   ├── test_embedder.py
│   ├── test_evaluation.py
│   ├── test_reranker.py
│   └── test_generator.py
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
       완료          완료         완료       완료          실험 구현
```

키워드 검색은 dense retrieval과 비교하기 위한 baseline입니다. 현재는 실제 LLM Reranker와 Generator를 연결하고, baseline과 재정렬 결과를 비교하며 실패 원인과 검증·재시도·fallback 동작을 살펴보는 단계입니다.

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

임베딩, LLM Reranker, 답변 생성 실험에는 `OPENAI_API_KEY`가 필요합니다. GitHub Codespaces에서는 저장소의 **Settings → Secrets and variables → Codespaces**에서 이름을 `OPENAI_API_KEY`로 등록한 뒤 Codespace를 다시 시작합니다. 로컬 셸에서는 현재 세션의 환경변수로 설정할 수 있습니다.

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

캐시를 생성한 뒤 다음 실험을 실행할 수 있습니다. 아래 스크립트는 기존 캐시를 읽으며 캐시를 직접 생성하지 않습니다. 질문 임베딩과 LLM 호출에는 API 비용이 발생할 수 있습니다.

```bash
# 대표 질문의 Retriever / Reranker 순위와 점수 비교 (단일 호출, 재시도 없음)
python -m scripts.real_reranker_experiment

# 동일한 평가 질문 3개로 Hit@3와 MRR 비교 (bounded retry 사용)
python -m scripts.reranker_evaluation

# Retriever Top 10 → Reranker Top 3 → 답변 생성 (RerankingError 시 fallback)
python -m scripts.run_rag
```

평가 스크립트는 순위 측정을 위해 재정렬된 Top 10 전체를 유지하고, 답변 생성 파이프라인은 최종 Top 3을 사용합니다. 기존 baseline 스크립트는 전체 청크 순위로 RR을 계산하고, 비교 평가 스크립트는 Top 10 밖의 정답을 미검색으로 처리합니다. 이번 기록의 정답은 모두 Top 10 안에 있었지만, 평가 질문을 늘릴 때는 이 차이를 고려해야 합니다.

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

테스트는 실제 OpenAI API를 호출하지 않습니다. 임베딩 결합 테스트는 가짜 임베딩 함수를 주입하고, dense retrieval 테스트는 고정 벡터를 사용합니다. OpenAI client가 모듈 import 시 생성되므로 테스트 실행 환경에도 `OPENAI_API_KEY`가 필요하지만, API를 호출하지 않는 테스트에는 임시 값으로 실행할 수 있습니다.

```bash
python -m pytest -v
```

테스트 범위에는 PDF 오류 처리와 텍스트 추출, 정규화, overlap 청킹과 페이지 추적, 업로드 API, 키워드 및 dense 검색, 입력 검증, 임베딩 결합의 불변성, Hit@k와 RR, 가짜 점수 reranking과 Generator prompt의 질문·본문·페이지 정보 구성이 포함됩니다. 현재 전체 회귀 테스트는 **35 passed**입니다. 이 결과는 실제 모델의 품질·안정성이나 validation·retry·fallback 전체에 대한 자동 검증을 의미하지 않습니다.

## 학습 기록

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

---

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

Day 3 당시 reranker는 실제 AI 모델을 호출하지 않았습니다. 후보와 가짜 점수를 결합하여 점수순으로 정렬하는 데이터 흐름만 먼저 검증했습니다. 이후 실제 모델 연결과 평가 내용은 아래 Reranker 실험 기록에 정리했습니다.

---

### Day 4: LLM Reranker 구현과 평가 (2026.09.16~17)

#### 문제와 관찰: 정답이 후보군에 있어도 낮은 순위에 위치함

Embedding Retriever만 사용했을 때와 LLM Reranker를 추가했을 때의 검색 품질을 비교했습니다. 위 Retrieval 평가와 동일한 3개 질문에 대해 relevant chunk를 직접 지정했습니다. 정답 청크는 질문 순서대로 `13`, `11`, `15`이며, baseline 수치는 위 Retrieval 평가에 기록했습니다.

대표적인 실패 질문은 다음과 같습니다.

> Which model does RAG use to retrieve passages from its document index?

정답인 chunk 13은 Retriever에서 8위였습니다. Hit@3은 `0`, RR은 `0.125`로, Top 10 후보군에는 포함되었지만 답변 생성에 사용하는 Top 3에는 들지 못했습니다. 이 문제를 해결하기 위해 Retriever Top 10을 GPT-5-nano에 전달하고, 각 후보가 질문에 직접 답하는 데 얼마나 유용한지 `0.0~1.0` relevance score로 평가하여 재정렬했습니다.

- **Retriever**: 의미적으로 관련 있는 후보를 넓게 검색합니다.
- **Reranker**: 후보 중 실제 질문에 답하기 좋은 청크를 재정렬합니다.
- **Generator**: 최종 선택된 청크를 근거로 답변을 생성합니다.

Retriever의 코사인 유사도와 Reranker의 relevance score는 의미가 다르므로 각각 `score`, `rerank_score`로 유지합니다. Reranker score는 prompt에서 정의한 관련도 점수이며, 보정된 확률(calibrated probability)이 아닙니다.

#### 원인 분석과 해결: 관련 단어와 직접 근거 구분

초기 실험에서는 chunk 13보다 단순히 `DPR`이라는 단어가 포함된 다른 청크에 높은 점수를 주는 경우가 있었습니다. chunk 13에는 DPR이 실제 retrieval component라는 직접 설명이 있었지만, 다른 청크에는 실험 결과 표 안에 DPR이라는 단어만 등장했습니다. 단어가 겹치는 것과 답변을 뒷받침하는 근거가 있는 것을 구분하도록 prompt에 다음 원칙을 추가했습니다.

```text
Do not give a high score merely because a candidate contains
keywords or terms related to the question.
Give a high score only when the candidate's context provides
evidence that directly supports an answer to the question.
```

#### Structured Output의 한계와 application level validation

JSON Schema 기반 Structured Output을 사용했지만, 올바른 JSON 구조가 모든 candidate를 정확히 한 번씩 평가했다는 뜻은 아니었습니다. 실제 Top 10 중 하나가 누락된 응답이 발생하여 application level에서 다음을 검증하도록 했습니다.

- 입력한 모든 candidate가 평가되었는가
- duplicate candidate가 존재하지 않는가
- 입력하지 않은 candidate가 반환되지 않았는가

`calculate_rerank_scores`는 임시 identifier mapping과 `validate_rerank_scores`를 통해 이를 확인합니다. `calculate_rerank_scores_with_retry`는 JSON 파싱·검증 관련 `ValueError` 발생 시 기본 최대 2회 재시도(최초 호출 포함 3회)하고, 반복 실패하면 `RerankingError`를 발생시킵니다. `scripts/run_rag.py`는 이 오류를 잡아 Retriever Top 3으로 fallback하여 답변 생성을 이어갑니다. 현재 이 처리는 모든 종류의 API 오류를 포괄하는 fallback은 아닙니다.

#### Identifier 충돌 분석과 LLM-facing ID 분리

Out-of-context 질문 실험에서는 입력하지 않은 chunk ID를 반환하는 문제가 반복되었습니다. 실제 prompt를 출력해 조사한 결과, 문서 References 영역의 `[50]`, `[51]`, `[52]` 같은 reference number를 숫자 기반 chunk identifier와 혼동하고 있었습니다.

내부 `chunk_index`를 candidate header에 직접 노출하는 대신, 호출마다 후보의 위치에 따라 임시 identifier를 부여하도록 변경했습니다. 다음은 mapping 예시입니다.

| Internal ID | LLM-facing ID |
| --- | --- |
| chunk 10 | `CANDIDATE_0` |
| chunk 27 | `CANDIDATE_1` |
| chunk 39 | `CANDIDATE_2` |

LLM은 `CANDIDATE_N`을 반환하고 Python에서 실제 `chunk_index`로 다시 mapping합니다. 제공된 실험 기록에서는 변경 후 반복되던 identifier 혼동이 해결되었으며, out-of-context 테스트에서도 첫 Reranker 호출이 validation을 통과했습니다. 이는 identifier 검증을 통과했다는 관찰이며, 문서 밖 질문에 대한 retrieval rejection을 구현했다는 뜻은 아닙니다.

#### 초기 3문항 재평가 결과

동일한 3개 질문으로 다시 평가한 초기 실행 결과는 다음과 같습니다.
아래 수치는 당시 실험 기록이며, README 갱신 과정에서 API를 다시 호출해
측정한 결과는 아닙니다.

| 지표 | Retriever | Retriever + Reranker |
| --- | ---: | ---: |
| Mean Hit@3 | 0.6667 | 1.0000 |
| MRR | 0.7083 | 1.0000 |

대표 질문의 relevant chunk 13은 **8위 → 1위**로 이동했고,
Hit@3은 **0 → 1**, RR은 **0.125 → 1.0**으로 개선되었습니다.

#### 15문항 확장 평가 결과

초기 실험 이후 평가 질문을 3개에서 15개로 확대했습니다.
직접 사실 확인, 개념 비교, 원인 설명, 실험 설정과 생성 방식 등
서로 다른 유형의 질문을 포함했습니다.

| 지표 | Retriever | Retriever + Reranker |
| --- | ---: | ---: |
| Mean Hit@3 | 0.8000 | 1.0000 |
| MRR | 0.6794 | 1.0000 |

15개의 질문 중 12개는 Retriever 단계부터 정답 청크가 Top 3 안에 있었습니다.
나머지 3개는 Retriever에서 Top 3 밖에 있었지만,
Reranker가 정답 청크를 모두 1위로 올렸습니다.

| 질문 유형 | Retriever 순위 | Reranker 순위 |
| --- | ---: | ---: |
| RAG가 사용하는 retrieval model | 8위 | 1위 |
| FEVER가 RAG 평가에 유용한 이유 | 5위 | 1위 |
| Jeopardy 생성에 token-level retrieval이 적합한 이유 | 5위 | 1위 |

각 평가 사례는 검색과 재정렬 결과에 따라 다음 outcome 중 하나로 분류합니다.

- `success`
- `retrieval_failure`
- `reranker_recovered`
- `reranker_regression`
- `unresolved_ranking_failure`

이번 실행에서는 `success`가 12건,
`reranker_recovered`가 3건이었습니다.

`failure_types`에는 outcome과 별도로
`retriever_top3_miss`, `reranking_failure`,
`reranking_regression`, `irrelevant_top_score` 같은
복수의 진단 신호를 기록합니다.

이를 통해 사례의 최종 결과와 세부적인 이상 신호를 함께 관찰할 수 있습니다.

#### 이번 실험에서 확인한 점

Retriever와 Reranker는 서로 다른 역할을 가집니다.
Retriever는 관련 가능성이 있는 후보를 넓게 찾고,
Reranker는 그 후보 중 질문에 직접 답하는 데 유용한 근거를 골라냅니다.

다만 Retriever가 정답 청크를 후보군에 포함하지 못하면
Reranker도 해당 청크를 복구할 수 없습니다.

Structured Output 역시 출력의 JSON 구조를 제한할 뿐,
모든 후보가 올바르게 평가되었다는 의미적 정확성까지 보장하지는 않았습니다.
따라서 LLM 출력에는 별도의 validation이 필요했습니다.

이번 구현에서는 다음 항목을 검사했습니다.

- 입력한 모든 candidate가 평가되었는가
- 같은 candidate가 중복으로 반환되지 않았는가
- 입력하지 않은 candidate가 반환되지 않았는가

검증 실패에 대비해 제한된 횟수의 retry를 적용하고,
반복 실패하면 Retriever 결과를 사용하는 fallback도 추가했습니다.

LLM-facing identifier와 내부 `chunk_index`를 분리하면서
문서 안의 reference number를 candidate identifier로 오인하는 문제도 줄였습니다.

이 과정을 통해 AI component를 사용하는 시스템에는
prompt뿐 아니라 validation, retry, fallback과 관찰 가능한 실행 기록이
함께 필요하다는 것을 확인했습니다.

현재 전체 회귀 테스트는 **42 passed**입니다.

## 학습 원칙

1. 전체 기능을 한꺼번에 구현하지 않습니다.
2. 각 기능이 필요한 이유를 이해한 뒤 코드를 작성합니다.
3. 파싱, 청킹, 검색과 생성처럼 서로 다른 책임을 분리합니다.
4. 정상 동작과 실패 상황을 함께 테스트합니다.
5. 입력, 출력, 목적과 trade-off를 직접 설명할 수 있는 상태를 목표로 합니다.
6. 실험 결과와 실패 사례를 기록하고 다음 구현의 근거로 사용합니다.

## 남은 한계와 확장 아이디어

현재 구현은 하나의 논문과 직접 작성한 15개의 평가 질문을 대상으로 합니다.
평가 결과 역시 한 번의 실행에서 얻은 값이므로,
다른 문서와 질문에서도 같은 성능이 나온다고 일반화할 수는 없습니다.

이번 프로젝트에서는 RAG의 각 단계를 직접 구현하고,
Retriever와 Reranker를 분리해 평가하는 것까지를 학습 범위로 정했습니다.

프로젝트를 더 확장한다면 다음 주제를 실험할 수 있습니다.

1. 평가 데이터와 질문 표현의 다양성을 확대합니다.
2. 반복 실행을 통해 Reranker 결과의 변동성을 측정합니다.
3. 모델별 품질, 비용과 지연시간을 비교합니다.
4. 문서 밖 질문에 대한 retrieval rejection을 구현하고 평가합니다.
5. LLM-as-a-Judge와 사람의 평가 결과를 비교합니다.
6. 캐시 fingerprint와 자동 무효화를 구현합니다.
7. 최종 답변의 페이지 인용이 실제 근거와 일치하는지 검증합니다.

이 항목들은 현재 프로젝트의 미완성 기능이라기보다,
향후 다른 프로젝트에서도 이어서 탐구할 수 있는
AI Engineering 주제입니다.

## 프로젝트 회고

### 왜 직접 구현했는가

LangChain이나 LlamaIndex 같은 프레임워크를 바로 사용하기보다,
RAG의 각 단계에서 데이터가 어떻게 변하는지 이해하고 싶었습니다.

그래서 PDF 파싱과 청킹부터 임베딩 검색, 재정렬과 답변 생성까지
각 단계를 하나씩 직접 구현했습니다.

처음에는 질문과 문서를 입력하면 답변을 생성하는 기능을 완성하는 것이
프로젝트의 중심이라고 생각했습니다. 하지만 개발을 진행하면서
AI Engineering에서 더 어려운 문제는 모델을 한 번 동작시키는 것이 아니라,
그 결과를 믿을 수 있는지 판단하고 계속 개선할 수 있는 환경을
만드는 것임을 배웠습니다.

### 정해진 답이 없는 시스템을 개발한다는 것

이번 프로젝트를 진행하며 AI Engineering에는
정해진 하나의 답이 없다는 점을 크게 느꼈습니다.

청크의 크기와 overlap, 검색 후보의 개수, Reranker의 평가 기준,
답변 생성에 사용할 문맥의 범위처럼 대부분의 선택에 trade-off가 있었습니다.

처음부터 완벽한 설정을 찾기보다 현재의 기준을 명확하게 세우고,
실험 결과를 관찰하면서 더 나은 선택을 반복해서 찾아가는 과정이
중요하다는 것을 배웠습니다.

### 예상대로 움직이지 않는 LLM

LLM은 일반적인 함수처럼 입력에 대해 항상 같은 형태와 품질의 출력을
보장하지 않았습니다.

프롬프트에 출력 형식을 명시해도 일부 candidate를 누락하거나,
예상하지 않은 identifier를 반환하거나,
관련 단어가 포함되었다는 이유만으로 잘못된 청크에
높은 점수를 주는 경우가 있었습니다.

자연어로 복잡한 작업을 지시할 수 있다는 점은 편리했지만,
사람에게 설명하듯 요청했다고 해서 항상 의도를 정확히 따르는 것은 아니었습니다.

이 경험을 통해 LLM을 활용하는 시스템에는 좋은 prompt뿐 아니라
출력 검증, 제한된 retry, fallback과 실패 기록이 필요하다는 것을 배웠습니다.

### 예상보다 훨씬 어려웠던 평가

『AI Engineering』을 읽으며 평가가 중요하고 어렵다는 사실은 알고 있었습니다.
하지만 직접 평가를 설계하기 전에는 그 어려움이 어느 정도인지
제대로 체감하지 못했습니다.

평가 지표를 계산하는 코드보다 더 어려웠던 것은
평가 데이터셋과 정답 기준을 만드는 일이었습니다.

어떤 질문이 실제 검색 능력을 확인하기에 적절한지,
어느 청크를 relevant chunk로 인정할지,
근거가 여러 청크에 흩어져 있을 때 어떻게 처리할지 직접 판단해야 했습니다.

최종 답변이 자연스럽다는 이유만으로
RAG 파이프라인 전체가 제대로 동작했다고 결론 내릴 수도 없었습니다.

Retriever가 올바른 근거를 찾았는지,
Reranker가 그 순위를 개선했는지,
Generator가 주어진 근거를 사용했는지를
각각 나누어 살펴봐야 했습니다.

책에서 읽을 때에는 평가가 하나의 개발 단계처럼 느껴졌지만,
직접 해보니 무엇을 성공으로 정의할지 결정하는 것부터가
평가의 일부였습니다.

### 개발자가 관찰할 수 있는 환경의 중요성

좋은 평가 기준을 만드는 것과 함께,
개발자가 모델의 판단 과정을 관찰할 수 있는 환경도 중요했습니다.

최종 출력만 남기면 문제가 발생했을 때 원인을 찾기 어렵습니다.
검색 순위, 코사인 유사도, Reranker 점수, relevant chunk와
최종 선택 결과를 함께 기록해야 어느 단계에서 품질이 떨어졌는지
추적할 수 있었습니다.

결국 AI Engineering에서 개발자의 역할은
모델에 요청을 보내는 데서 끝나지 않는다고 느꼈습니다.

평가 기준을 만들고, 결과를 기록하고, 실패를 분류하고,
그 근거를 사람이 확인할 수 있도록 시스템을 설계하는 일까지
개발자의 역할에 포함됩니다.

이번 프로젝트를 통해 RAG의 기본 구조를 직접 구현하고,
각 단계를 분리해 평가하는 경험을 얻었습니다.

다음 프로젝트에서는 실시간 입력, 상태 관리, 지연시간과 사용자 피드백처럼
새로운 AI Engineering 문제를 다뤄보고자 합니다.