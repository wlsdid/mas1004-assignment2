# Log

One entry for each recipe, selection or Track B model you train, newest at
the bottom. The numbers come from `experiments.csv`. "Why" and "what it means"
are in your own words: your agent asks you for them and does not make them up.
End the log with your plan for what to try next.

<!-- The form of an entry:

## <date> <tag> (Track A or B)
- changed: what is different from the run before
- why: your reason
- validation: 0.000 (before: 0.000)
- what it means: what you think the result says
-->

## 2026-11-02 baseline (Track A)
- changed: 스타터 레시피 그대로, 시드 0과 1로 두 번 학습 (Colab)
- why: (아직 적지 않음)
- validation: 시드 0 0.4956, 시드 1 0.4942
  (시드 0: seoul 0.574, busan 0.368, tokyo 0.632, osaka 0.345, taipei 0.559;
  시드 1: seoul 0.555, busan 0.402, tokyo 0.619, osaka 0.371, taipei 0.524)
- what it means: (아직 적지 않음)

### baseline 시드 0의 오답 시트에서 본 것
`results/curate/mistakes_baseline_seed0_<city>_00.png` 다섯 장을 보고 적은 내용.

- 틀린 사진 중에는 도로나 길을 찍은 사진이 많다.
- 사진 속 글자가 잘 보이지 않는다.
- 자동차나 비행기 같은 사물이 많이 찍혀 있다.
- 특색이 없어 어디서나 볼 수 있는 음식 사진이 섞여 있다.

## 2026-11-05 clip8000 (Track A)
- changed: 학습 사진 선택만 바꿨다. 레시피는 그대로다.
  - 규칙: 글씨나 건물이 보이지 않는 도로 사진, 사물만 있거나 인물만 있는 사진은 뺀다.
    음식 사진 중 카페나 브런치처럼 어디서나 볼 수 있는 음식은 빼고, 그 지역의 전통
    음식이나 유명한 음식은 남긴다.
  - 부산 샘플 시트(look_busan_00~02)에서 이 규칙에 맞는 사진 20장을 직접 골라 뺐다.
  - 나머지는 CLIP으로 학습 사진마다 규칙에 맞는 정도를 점수로 매기고
    (`scripts/clip_filter.py`), 순위 구간별 시트를 보고 8,000위까지 빼기로 정했다.
    그 안의 음식 사진은 지역 음식 점수 0.75 이상이거나 제목에 음식 이름이 있으면 남겼다
    (649장).
  - 학습 사진 44,257장 → 36,886장 (seoul 9,516, busan 4,194, tokyo 11,647,
    osaka 6,074, taipei 5,455)
- why: AI가 사진을 분류할 기준을 명확히 해주고 싶었다. 카페나 브런치 같은 음식은
  어디서나 볼 수 있어서 지역을 구분할 단서가 되지 않는다. 차나 도로도 특색이 있는 것이
  아니라면 사람도 직관으로 구분하기 어렵다. 그래서 이런 사진은 AI의 학습에도 방해가 되고,
  AI가 잘못된 규칙을 배울 수 있다고 생각했다. 반면 건물 양식이나 표지판의 글자로 지역을
  유추할 수 있는 사진은 남겼다. 음식도 그 지역을 상징하는 음식은 남겼다. 물론 어떤 음식이든
  다른 곳에서도 볼 수 있지만, AI는 확률로 판단하므로 특정 지역 음식을 그 지역으로 높은
  확률로 판단하는 것은 괜찮다고 생각했다.
- validation: 0.4917 (before: baseline 0.4956, 시드 1 0.4942)
  (seoul 0.567, busan 0.378, tokyo 0.632, osaka 0.378, taipei 0.504)
- what it means: (결과를 보고 적는다)
