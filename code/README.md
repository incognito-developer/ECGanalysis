# Introduction
여기는 학습 코드와 학습된 모델을 다운받을 수 있는 디렉토리입니다.

basic 모델은 CNN + LSTM 을 이용해서 만든 모델입니다.

basicWithDropout 모델은 CNN + LSTM을 기반으로, Dropout을 사용한 모델입니다.

useEssemble은 Essemble을 이용한 실험적인 모델입니다. MITBIH dataset과 PTBDB dataset를 가지고 train과 test를 서로 다른 데이터 종류로 하면 학습이 제대로 안된다는 것을 발견했습니다. 그래서 basicWithDropout 모델을 가지고, MITBIH dataset으로 학습시킨 모델과 PTBDB dataset으로 학습시킨 모델을 enssemble 한 후, score을 평균내는 방식으로 해보았습니다. 좋은 결과는 나오질 않습니다. 

transformerModel은 multihead attention을 이용한 transformer를 적용시킨 모델입니다.
