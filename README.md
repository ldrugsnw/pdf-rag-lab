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
