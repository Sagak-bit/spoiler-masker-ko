# 스포일러 가리개 (웹 데모)

리뷰에서 스포일러 문장만 골라 가리는 한국어 모델 CLOAK(KoELECTRA + 리뷰 BiGRU)를 브라우저에서 바로 돌려 보는 정적 사이트.
CAS2250 인공지능개론 팀 프로젝트의 결과물이며, 이 저장소는 `final/webdemo/build_static.py`가 만든 빌드 결과다.

- 서버가 없다. [스포일러 가리기]를 처음 누르면 모델(CLOAK 약 124 MB, 비교용 '문장 하나만 보는 모델' 약 113 MB)을 Hugging Face
  (`https://huggingface.co/dPckslim/koelectra-spoiler-ko/resolve/main/onnx/`)에서 한 번 받아 브라우저에 저장하고, 추론은 onnxruntime-web(WebAssembly)으로 이 브라우저 안에서 한다.
  입력한 글은 어디로도 보내지 않는다.
- 모델 파일은 가중치만 int8로 줄인 것이다(계산은 fp32). test 리뷰 400개(4,909문장)에서 원래 모델과 가림 판단이
  다른 문장은 Precision 우선 기준 3개, Recall 우선 기준 13개였고 PR-AUC는 0.3864 → 0.3860이었다.
- 직접 입력한 리뷰의 DF-IIF 특징은 서버 데모(Kiwi 형태소 분석)와 달리 IIF 어휘에서 가장 긴 접두어를 찾는 방식으로
  명사를 고른다. test 리뷰 150개(1,845문장)에서 Kiwi로 계산한 경우와 판단이 다른 문장은 Precision 우선 3개, Recall 우선 20개였다.
- 예시 리뷰와 "리뷰 사이트에서 보기"의 결과는 test 평가와 같은 입력으로 미리 계산한 값이다.
- 리뷰는 Goodreads 리뷰(UCSD Book Graph)를 한국어로 기계 번역한 것이고, 글꼴은 Pretendard(OFL, `static/fonts/NOTICE.txt`)다.
