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
