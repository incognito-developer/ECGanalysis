This models are trained by MITBIH Dataset, PTBDB Dataset each, not combined Dataset.
Then ensembled the result.
Since there are only two models, the voting method was not used, and the average using numpy.mean() was used.
reason: How to classification different Two dataset without combine dataset train.

result: failed 

이모델들은 MITBIH 데이터셋과 PTBDB 데이터셋을 합치지 않고 각각 학습 시킨 모델입니다.
이 두 모델들을 가지고 essemble을 이용해 결과를 측정해봤습니다.
모델이 두개뿐이라 투표방법을 이용하지 않고, numpy.mean()을 이용한 평균을 이용했습니다.
이유: 어떻게 하면 두 데이터셋을 섞어서 학습하지 않고, 다른 두 종류의 데이터셋을 구분할 수 있을까?

결과: 실패
