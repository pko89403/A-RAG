---
name: research-paper-comparison
description: 두 개 이상의 논문을 선정하여 핵심 기준(기여, 방법론, 데이터셋, 실험결과 등)별로 상세 비교 분석한다.
---

## 공통 도구 계약(필수)

- **철학 준수**: "검색은 힌트(신호), 독서는 확정(사실)" 원칙을 따른다.
- **Search**: `keyword_search` 또는 `semantic_hybrid_search`를 사용하여 비교 대상 논문의 ID를 확보하기 위한 '후보 신호'를 찾는다. 스니펫 내용을 비교 근거로 사용하지 않는다.
- **Read**: 비교하려는 각 논문의 핵심 섹션(Abstract, Method, Experiments)을 반드시 `chunk_read`로 정독한다.
- **Think**: `research_paper_think`를 통해 비교 기준(Criteria)을 수립하고, 각 논문에서 해당 내용이 추출되었는지(Fact Check) 교차 검증한다.
- **Answer**: 최종 비교표는 `chunk_read`로 확인된 사실에 기반하여 작성하며, `cite_sources`로 근거를 명시한다.

## 언제 사용하나(트리거)

- "A 논문과 B 논문 비교해줘", "두 모델의 차이점은?", "성능 비교표 만들어줘"
- 복수의 논문을 특정 기준으로 대조 분석해야 할 때

## 목표(산출물)

- **비교 분석 테이블**: 논문들을 행(Row)으로, 비교 기준을 열(Column)로 하는 마크다운 테이블 작성.
- **차이점 요약**: 방법론적 차이, 성능 차이, 접근 방식의 철학적 차이 등을 서술.
- **우위 분석**: 각 논문의 장단점 및 특정 상황에서의 우위를 분석.

## 권장 워크플로

1) **논문 식별**: 사용자 질문에서 비교할 논문들을 식별하고, 각각 검색(`keyword_search`)하여 ID를 확보한다.
2) **기준 수립**: 비교할 기준(예: Model Architecture, Training Data, Pre-training Objective, Fine-tuning Method, Evaluation Metrics, SOTA Comparison)을 정의한다.
3) **내용 추출(`chunk_read`)**: 각 논문의 Abstract, Method, Experiments 섹션을 중점적으로 읽어 기준별 내용을 추출한다.
4) **교차 검증**: 한 논문에는 있는데 다른 논문에는 없는 정보가 있는지 확인하고, 필요 시 추가 읽기를 수행한다.
5) **비교표 작성**: 수집된 정보를 바탕으로 비교표를 작성한다. 정량적 수치(Accuracy, F1-score 등)는 정확하게 기재한다.
6) **종합 분석**: 단순 나열을 넘어, 왜 성능 차이가 발생했는지, 어떤 접근법이 더 효율적인지 분석 의견을 제시한다.
