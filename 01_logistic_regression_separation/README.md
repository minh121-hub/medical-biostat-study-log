# Clinical Logistic Regression: 완전분리 문제와 해결 (Perfect Separation)

## 배경 (Background)

로지스틱 회귀(logistic regression)는 임상 예측 모델에서 가장 널리 쓰이는 기법 중 하나지만, 실무에서는 데이터에 실수로 결과변수(outcome)와 거의 동일한 정보를 가진 변수가 섞여 들어가는 경우가 있다 (data leakage). 이 경우 완전분리(perfect separation)라는 통계적 문제가 발생해 모델 적합(fit) 자체가 실패하거나 계수가 발산한다.

이 프로젝트는 완전분리 문제를 의도적으로 재현하고, 이를 진단·해결하는 전체 과정을 다룬다. 목표는 단순히 로지스틱 회귀를 "돌리는" 것이 아니라, 모델이 깨지는 상황을 인지하고 원인을 좁혀나가는 진단 과정을 보여주는 것이다.

## 데이터 (Data)

n=150의 합성(synthetic) 임상 데이터. 실제 환자 데이터가 아니며, 재현 가능성을 위해 랜덤 시드를 고정했다.

| 변수 | 타입 | 설명 |
|---|---|---|
| age | 연속형 | 30~80세, 정규분포 근사 (mean=55, sd=12) |
| bmi | 연속형 | 18~40, 정규분포 (mean=26, sd=4) |
| smoking | 범주형(0/1) | 흡연 여부, p=0.3 |
| systolic_bp | 연속형 | 90~180mmHg, 정규분포 (mean=130, sd=15) |
| outcome | 이진형(0/1) | age, bmi, smoking, systolic_bp의 선형결합에 로지스틱 함수를 적용해 생성한 확률적 이진 결과 |
| separator | 이진형(0/1) | **인위적으로 삽입한 변수.** outcome을 그대로 복사한 값 (일부 노이즈 포함 가능). 실수로 결과변수와 거의 동일한 정보가 모델에 섞여 들어간 상황(data leakage)을 재현하기 위한 변수 |

## 방법론 (Methodology)

1. **기본 데이터 생성**: age, bmi, smoking, systolic_bp로 선형결합(linear combination)을 만들고 sigmoid 함수로 확률 변환, 이항분포로 outcome 샘플링
2. **완전분리 삽입**: separator 변수를 outcome과 거의 동일하게 생성해 X에 포함
3. **모델 적합 시도**: `statsmodels.Logit`으로 age, bmi, smoking, systolic_bp, separator 전체를 포함해 적합
4. **1차 진단 — VIF**: 다중공선성(multicollinearity) 여부 확인. 다중공선성과 완전분리는 서로 다른 문제이므로, VIF가 정상이어도 완전분리를 배제할 수 없음을 확인하는 단계
5. **2차 진단 — 상관관계 스캔**: 각 독립변수와 outcome 간 상관계수(`corrwith`)를 전부 계산해 원인 변수를 특정
6. **해결**: 원인 변수(separator) 제거 후 재적합
7. **재적합 결과 확인 및 임상적 해석**

## 결과 (Results)

**separator 포함 시**: `PerfectSeparationWarning` 경고 발생, 로그우도(log-likelihood)가 발산(`inf`), 최종적으로 `LinAlgError: Singular matrix`로 모델 적합 실패.

**VIF 진단**: age, bmi, smoking, systolic_bp, separator 전부 VIF 1.0~1.1대로 다중공선성 없음. → 다중공선성은 문제의 원인이 아님을 확인.

**상관관계 스캔**: age(0.121), bmi(-0.042), smoking(0.152), systolic_bp(0.133)는 평범한 수준인 반면, **separator는 outcome과 상관계수 1.0**으로 원인 특정.

**separator 제거 후 재적합**: 정상 수렴(`converged: True`), 계수 및 표준오차 모두 안정적인 범위로 복원.

| 변수 | 계수 | Odds Ratio | p-value |
|---|---|---|---|
| age | 0.0302 | 1.031 | 0.065 |
| bmi | -0.0533 | 0.948 | 0.249 |
| smoking | 0.9165 | 2.500 | 0.017 |
| systolic_bp | 0.0294 | 1.030 | 0.018 |

## 해석 (Interpretation)

- **smoking**: 오즈비(Odds Ratio) 약 2.5, 통계적으로 유의(p=0.017). 흡연자는 비흡연자 대비 outcome 발생 오즈가 약 2.5배 높다. 이 모델에서 가장 뚜렷한 예측 인자.
- **systolic_bp**: 오즈비 약 1.03로 작아 보이지만, 1mmHg 단위가 아니라 10mmHg 차이로 환산하면 오즈가 약 34% 차이 나는 수준이라 임상적으로 무시할 수 없다.
- **age**: p=0.065로 유의수준 0.05에 근접했으나 도달하지 못함 (경계선적 결과).
- **bmi**: 이 데이터에서는 유의한 연관성을 보이지 않음.

**핵심 메시지**: VIF는 독립변수들 사이의 다중공선성을 정확히 진단하는 도구이지만, 완전분리(독립변수와 종속변수 사이의 문제)는 애초에 VIF의 진단 범위 밖이다. VIF가 정상으로 나왔다고 해서 모델에 문제가 없다고 판단해서는 안 되며, 완전분리는 수렴 경고, 계수 발산, 특이행렬(singular matrix) 에러 등 별도의 신호로 확인해야 한다.

## 파일 구조

- `notebook.ipynb`: 데이터 생성부터 최종 해석까지 전체 분석 코드
- `README.md`: 본 문서

## 배운 점 (Lessons Learned)

- 완전분리와 다중공선성은 서로 다른 문제이며, 각각 다른 진단 도구가 필요하다는 것을 직접 확인했다.
- 처음에는 특정 변수(systolic_bp)의 극단값으로 완전분리를 재현하려 했으나, 신호가 약해 실패했다. 이를 통해 완전분리가 실제로 발생하려면 해당 변수가 전체 데이터 패턴에서 무시할 수 없을 만큼 강한 결정력을 가져야 한다는 것을 체감했다.
- 진단 순서(모델 실패 확인 → VIF → 상관관계 스캔 → 원인 제거)를 실제로 밟아보며, 통계 도구 하나에만 의존하지 않고 여러 각도에서 검증하는 것이 중요하다는 것을 배웠다.
