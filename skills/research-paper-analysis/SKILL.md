---
name: research-paper-analysis
description: 논문의 핵심 기여, 방법론, 실험 결과, 한계점 등을 구조적으로 심층 분석한다.
---

## 공통 도구 계약(필수)

- **철학 준수**: "검색은 힌트(신호), 독서는 확정(사실)" 원칙을 엄격히 따른다.
- **Search Tools (`semantic_hybrid_search`, `contextual_search`, `keyword_search`)**:
  - 역할: 정답의 실마리가 될 '후보 신호(Candidate Signal)'를 탐색한다.
  - `contextual_search`: 논문의 복잡한 구조(페이지/표/그림)를 고려한 맥락 검색에 유리하다.
  - 제약: 반환된 스니펫은 불완전하므로, 절대 이를 바탕으로 최종 답변을 확정하거나 사실로 간주하지 않는다.
- **Read Tool (`chunk_read`)**:
  - 역할: 검색된 신호를 바탕으로 원문 전체를 확인하는 **심층 분석(Deep Analysis)** 도구.
  - 규칙: 모든 사실 관계 확인과 분석은 반드시 `chunk_read`로 확보한 Full Text를 기반으로 수행한다.
- **Think Tool (`research_paper_think`)**:
  - 매 Search와 Read 사이, 그리고 답변 전에 호출하여 "스니펫만으로 추측하지 않았는지", "확보한 근거가 충분한지" 점검한다.
- **Workflow**: `Search (Find Hint)` → `Think (Select Candidates)` → `Read (Confirm Facts)` → `Answer (Cite Sources)`.
- `cite_sources` 호출 시 이번 턴에서 `chunk_read`로 읽은 id만 전달한다.
- 쿼리 재작성: 검색 결과가 0건이면 "근거 부족" 선언 전 반드시 쿼리를 재작성(동의어, 영문 등)하여 재시도한다.

## 언제 사용하나(트리거)

- "논문 분석해줘", "핵심 내용 요약해줘", "이 논문의 한계점은?", "실험 결과 비교해줘" 요청
- 특정 논문에 대한 심층 분석이 필요할 때

## 목표(산출물)

- 논문의 구조적 핵심 내용을 마크다운 리포트 형태로 정리한다.
- 핵심 기여, 방법론, 실험 결과, 한계점을 명확히 구분하여 제시한다.
- 모든 주장은 원문 인용(`<ref N> <page: P>`)으로 뒷받침되어야 한다.

## 권장 워크플로

1) 대상 논문을 검색(`keyword_search` 또는 `semantic_hybrid_search`)으로 찾고 `chunk_read`로 주요 섹션(Abstract, Introduction, Method, Experiments, Conclusion)을 읽는다.
2) 핵심 기여(Core Contribution): 기존 연구 대비 차별점과 독창성을 파악한다.
3) 방법론(Methodology): 제안하는 모델/알고리즘의 동작 원리를 파악한다.
4) 실험 및 결과(Experiments & Results): 실험 설정(Dataset, Baseline)과 주요 정량적 성과(Metric)를 확인한다.
5) 한계점(Limitations): 저자가 언급한 한계점이나 실험 결과의 제약 사항을 파악한다.
6) 근거가 부족하면 추정하지 말고 "근거 부족"으로 표시하거나 추가 검색을 수행한다.
7) 최종 리포트는 `RESEARCH_PAPER_SUMMARY_AGENTS.md`의 답변 포맷을 준수하여 작성한다.
