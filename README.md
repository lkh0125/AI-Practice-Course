# AI실무인증과정 
---------
> #### AI 실무능력인증과정은 현업에서 바로 활동할 수 있는 AI 실무전문가를 양성하기 위해 국내 최초(2013년 개설)의 AI빅데이터 분야 대학원 석사과정을 운영중인 국민대학교 경영대학원과 우리나라 지능형 정보시스템의 발전을 선도해 온 한국지능정보시스템학회가 공동으로 개설한 과정
---------
---------

## `1.주제 선정 배경`
- `연구동기 및 중요성`
  - 미래의 여러 Time Step에 대해서 예측을 진행하는 Multi-Horizon Forecasting은 시계열 나아가 예측 분야에서 매우 중요한 문제로 과거의 전통통계분석 기법에서 기존에 발견하지 못한 패턴을 찾아낼 수 있는 딥러닝 시계열 모델 연구가 활발히 이루어 지고 있습니다. <br> &nbsp; 특히 이러한 연구중에 여러가지 다양한 입력변수들을 통해 딥러닝 모델에 다변량 변수가 미치는 영향도를 파악하려는 연구들도 활발히 이루어지고 있는데 유통업 헬스케어 금융분야등에서 중요하게 사용되고 있으며 이는 판매량 예측과 같은 비즈니스 결정과, 가격예측, 재고관리 최적화등 의사 결정을에서 내리는데 매우 중요하게 사용되고 있습니다.<br>
- `수집데이터`
  - 가상화폐의 일자별 종가데이터와, 거래량, 저가, 고가, 변동가, 변동율 등 2017년9월25일부터 2022년5월28일까지 대략 5년 동안의 가상화폐의 거래 지표를 수집
  - 세계 주요 6개국 통화인 일본 엔, 유로, 영구 파운드, 스위스 프랑, 캐나다 달러, 스웨덴 크로네에 대한 미 달러 환율 및 위안/달러, 달러/원화 환율
  - 검색 트래픽 데이터인 구글 트랜드 지수
  - 그래픽 카드의 가격을 결정짓는 핵심 부품인 GPU, RAM을 생산하는 기업의 주가(엔비디아(NVIDA), AMD, 삼성전자, 하이닉스)
  - 두바이유, 브랜트유, WTI(서부 텍스트유)등의 배럴당 가격
  - 금을 비롯한 원자재 등의 실물 자산 대체재 가격
  - 옥수수, 커피, 코코아등 곡물 시장 가격
  - 심리적 요인을 반영한 공포 탐욕 지수
  - 한국과 미국의 공휴일 (원핫 인코딩)
  - 암호화폐 가격의 이동평균 가격과, log변환, 주별, 월별 평균가격을 가공
  - 크롤링으로 네이버 뉴스 데이터를 수집해 문서내에 어떤 주제가 들어있고, 주제 간의 비중이 어떤지 문서 집합 내의 단어 통계를 수학적으로 분석하는 LDA 주제 분류 모형(잠재 디리클레 할당, Latent Dirichlet Allocation)을 활용하여 분류된 주제의 지표값을 범주화한뒤 학습데이터에 추가
  - 전쟁이나 금리인하와 같은 특정 사건에 대해 화폐 가격이 움직이는 경우를 생각하여 수집한 기사에서 korean KeyBERT 모델을 통해 기사의 핵심 키워드를 추출, 지표화하여 학습데이터에 추가<br>
- `데이터 셋`
  - 수집된 데이터를 있는 그대로 대부분 활용하는 데이터 셋.
  - 여러 변수들중 서로 상관성이 높은 변수들의 선형결합으로 이루어진 주성분이라는새로운 변수를 만들어 변수들을 요약하고 축소하는 PCA 기법을 활용하여 Scree plot을 통해 분산 설명력이 80이 넘는 주성분 2개로 축소하고 시계열 모델 input에 필요한 주, 월 등의 범주지표를 결합하는 데이터 셋.
  - Temporal Fusion Transfomers에서 입력되는 time_varying_unknown_reals 값의변수 유형에 따라성능의 편차가 큰것을 확인하고 기존에 수집한 데이터를 F-통계량과 AIC 값을 활용하여 단계적 변수 선택법을 통해 유의미한 독립 변수만을 선택한 데이터 셋.<br><br>
## `2.문제 정의 및 가정`
- `연구 목적`
  - 가상 화폐 가격을 예측하기 위해서 시계열 예측 모형들을 이용하여 다양한 입력 변수들이 예측 성능에 미치는 영향도를 파악.
- `연구 방법`
  - Baseline, DeepAR, TFT 실험과정으로 찾은 최적의 하이퍼 파라미터의 조합으로 시계열 모델을 구축
    - RNN 기반의 LSTM 모델(Baseline)
    - RNN 기반의 확률적 예측 모형인 DeepAR
    - Attention 기반의 구조 모형인 Temporal Fusion Transformers
- `성능 지표`
  - 백분율 오류를 기반으로 한 정확도 측정 방법인 SMAPE(Symmetric mean absolute percentage error)사용
- `성능 향상 방법`
  - 암호화폐와 관련된 뉴스 데이터를 이용하여 마이닝 기법을 일부 적용하여 비정형 데이터를 공변량으로 활용하고 분석에 가치가 있는 항목들에 대한 정보를 수집한 후 적절한 통계학적 절차를 통해 필수적인 공변량을 선택하고 딥러닝 시계열 모델에 반영하여 암호 화폐 예측을 위한 공변량의 효과를 파악<br><br>
## `3.제안 방법론 - 모델 파이프 라인`
- `DeepAR`
  - [DeepAR](./Image/DeepAR.jpg) 
  - Auto-regressive recurrent network model을 기반으로 하는 확률 모형으로 미래의 값이 아니라 미래 확률 분포를 추정하며 여러 공변량을 활용해 학습이 가능한 모델
   
  - 단일 예측값이 아닌 해당 시점의 확률 분포를 모델링하는 파라미터가 output으로 출력되어 probabilistic forecasting
   
  - 일반 RNN과의 가장 큰 차이중 하나는 Likelihood Model을 사용한다는 것으로 확률값으로 나오는 실제 데이터의 분포를 찾기 위해 Gaussian Likelihood와 건수등에 대해서는 Negative binomial distribution를 사용하여 정확도를 높여줌
   
  - 보통 RNN은 이전 스텝에서의 출력 값을 현재 스텝의 입력값으로 사용하는데 DeepAR은 정확한 데이터로 훈련하기 위해 예측값을 다음 스텝으로 넘기는 것이 아니라 실제값을 매번 입력값으로 사용하는 교사강요(teacher forcing) 방식으로 훈련
   
  - covariate 과 함께 학습하기 때문에 시계열 패턴뿐만 아니라 예를들어 블랙프라이데이에 첫 수요가 발생했다면 다음 블랙프라이데이에도 수요가 발생할 것이라 예측하고, 해당 품목의 재고를 준비할 수 있음
   
  - 신제품의 수요를 예측할 때 발생하는 cold-start 문제를 비슷한 제품의 수요 데이터를 활용하여 예측에 활용

- `Temporal Fusion Transformers`
  - 미래에는 알 수 없는 관측 변수(Observed Inputs)들과 함께 알고 있는 변수(Time varying Known Input), 시간에 따라 변하지 않는 변수인 Static Covariates을 입력으로 활용하여 Multi-Horizon Forecasting을 하는 Attention기반 구조로 구성
   
  - 단일 예측값이 아닌 해당 시점의 확률 분포를 모델링하는 파라미터가 output으로 출력되어 probabilistic forecasting
  - Encoder Variable 그리고 Decoder Variable 간의 Feature Importance 제공하여 모델의 해석력을 확보할 수 있다는 장점이 있는 모델
  - Gating Mechanism은 불필요한 성분을 스킵하여 광범위한 데이터셋에 대하여 깊이와 복잡도를 조절하는 기능
  - GRN(Gated Residual Network)는 Input Variable과  선택적으로 context vector를 받아 처리하고 ELU activation function(Exponential Linear Unit Activation Function : ELU는 입력이 0보다 크면 Identify Function으로 동작하며, 0보다 작을 때는 일정한 출력) 을 거친 후 GLU(Gated Linear Units)을 사용하여 주어진 데이터셋에서 필요하지 않은 부분을 제거한 후 Variable Selection Network를 통해 관련 있는 Input Variable만 선택
  - Static Covariate Encoder를 통해 Static Covariate들을 Context Vector에 인코딩하고 네트워크에 연결
  - Temporal Processing을 통해 현재는 관측할 수 있으나 미래는 알수 없는 값과 미래에도 알수 있는값 모두에 대해 장기 단기 시간 관계를 학습
  - Interpretable Multi-Head Attention을 통해 장기 의존성 및 해석력을 강화하는데 TFT는 Self-Attention Mechanism을 통해 다른 시점에 대한 Long-Term Relation을 파악하는데 각 Head에서 출력된 값을 공유하기위해 기존의 Multi-Head Attention 구조와는 조금 다르게 수정된 Multi-Head Attention 구조를 사용
  - Prediction Intervals을 통해 Quantile을 이용하여 매 Prediction Horizon에 대하여 Target이 존재할 수 있는 범위를 제공<br><br>

## `4.실험 및 결과 - 실험 시나리오 설계`
  - PCA 차원 축소 데이터셋, 회귀 방정식으로 선택된 설명변수들로 구성된 데이터셋 그리고 전체 데이터셋에 대해 각각 Baseline인 LSTM, DeepAR, Temporal Fusion Tranformers 모델에 적용하여 백분율 오류에 기반한 정확도 측정 평가지표인 SMAPE값을 비교, 모델의 성능을 측정
   
   $$ SMAPE = 100/n * \displaystyle\sum_{i=1}^{n}\vert y_l - \widehat{y_l} \vert / (\vert y_l \vert +  \vert \widehat{y_l} \vert) $$

  - 해석이 가능한 Temporal Fusion Transformers으로 타겟에 영향을 주는 변수의 중요도도 확인
  - 효율적으로 하이퍼파라미터를 탐색하기 위해 TPE (Tree-structured Parzen Estimator) 알고리즘을 사용
   
    - TPE (Tree-structured Parzen Estimator) 알고리즘
       > 미지의 목적 함수의 대한 확률적인 추정을 수행하는 Surrogate Model 모델과 목적 함수에 대한 현재까지의 확률적 추정 결과를 바탕으로 최적의 입력값을 찾는 데 가장 유용할 만한 다음 입력값 후보을 추천해 주는 함수인 Acquisition Function이 존재해서 미지의 목적 함수(objective function) 의 최적의 해를 찾는 Bayesian Optimization 방법론

<br>

## `5.실험 결론`
  - PCA 차원축소 또는 변수 선택법을 통해 변수를 축소하여 사용하는것보다 다양한 input 데이터를 직접 사용하는 것이 DeepAR, TFT 둘다 성능이 비교적 좋았으며 예측값에 대해 멀티스탭으로 구현시 예측결과는 DeepAR, TFT가 Baseline에 비하여 우수한 성능을 보여주는것을 확인
   
  - TFT의 Feature importance를 통해 변수들의 중요도를 파악한 결과를 보여주는데 미래 시점의 값을 현재 시점에서도 알 수 있는 변수(time_varying_known_categoricals) 인공휴일, 요일, 월 등이 데이터가 예측에 영향을 주고 있는것을 확인
   
  - 달러/원화 환율, 달러/위안 환율, 등이 영향을 주고있는 것을 확인할수 있었는데 특이한 점은 예측기간이 짧기는 하지만 왼쪽에 달러/위안 환율의 예측값이 실제 화폐 가격값과 반대의 성향을 띄고 있음
   
  - DeepAR도 추세는 실제 화폐 가격에 따라가고 있으나 예측 정확도는 TFT에 비해 비교적 낮았고 멀티스탭으로 구현한 LSTM보다는 성능이 우수함을 보여줌
   
  - 멀티스탭으로 구현한 LSTM의 경우 예측 기간이 길어질수록 예측력이 떨어지는 것을 볼수 있음

<br>
