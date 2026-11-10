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
- why: 과제 안내대로, 시드만 바꿨을 때 점수가 우연히 얼마나 움직이는지 보려고
- validation: 시드 0 0.4956, 시드 1 0.4942
  (시드 0: seoul 0.574, busan 0.368, tokyo 0.632, osaka 0.345, taipei 0.559;
  시드 1: seoul 0.555, busan 0.402, tokyo 0.619, osaka 0.371, taipei 0.524)
- what it means: 학습 점수가 전부가 아니고, 점수에는 우연이 섞여 있다. 많은 학습으로 점수를 안정화하는 것이 필요하다.

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
    (`scripts/clip_filter.py`), 순위 구간별 시트를 보고 8,000위까지 빼기로 정했다. 8,000위 위로는 남길 만한 사진이
    눈에 띄지 않았고, 그 아래로는 주변의 건물이나 표지판이 조금씩 보이기 시작했다.
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
- what it means: 전체 학습량에 비해 빠진 사진이 너무 적었다. 또 여러 지역에서 골고루 빠진 것이 아니어서 학습에 긍정적인 영향이 적었다고 생각한다.

## 2026-11-09 clip8000_noflip (Track A)
- changed: clip8000과 같은 사진 선택에서, 레시피의 좌우 뒤집기(`T.RandomHorizontalFlip()`)만 뺐다.
  이 컴퓨터의 CPU로 학습했다 (115분).
- why: 좌우 뒤집기를 하면 통행 방향도 바뀌고 글자도 뒤집혀 알아볼 수 없다. 이러한 정보가
  판단에 영향을 주는지 보고 싶었다.
- validation: 0.4953 (before: clip8000 0.4917)
  (seoul 0.594, busan 0.393, tokyo 0.618, osaka 0.367, taipei 0.505)
- what it means: 점수의 큰 차이가 없는 것을 보아 큰 영향이 없는 것으로 추측할 수 있었다.

## 2026-11-10 b3 (Track B)
- changed: b1(ResNet18, 224, 스타터 사진)에서 두 가지를 바꿨다.
  - 모델: EfficientNet-B3, 사진 크기 320 (10.66M 파라미터, 1.96 GMACs), 8에포크,
    도시 균등하게 뽑기, 좌우 뒤집기 없음
  - 사진: 스타터 학습 사진에 Mapillary에서 직접 모은 거리 사진 1,894장(`data/mine`)을 더했다
    (46,151장).
  - Colab에서 학습했다. 같은 설정이 두 번 돌았다 (코드만 바뀌어 다시 실행됨).
- why: 같은 나라의 도시를 구별하는 데 거리 모습이 도움이 될 것 같아서, 양질의 데이터를 모으기
  위해 직접 수집했다. 또 더 높은 성능의 모델을 사용하는 것이 도움이 될 것 같았다.
- validation: 0.5901, 0.591 (before: b1 0.4832)
  (두 번째: seoul 0.609, busan 0.566, tokyo 0.610, osaka 0.506, taipei 0.663)
- what it means: 점수가 크게 오른 것을 보고 좋은 데이터를 수집하는 것이 중요하다고 생각했다.
