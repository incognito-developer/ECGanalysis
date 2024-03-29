# ECGanalysis
to practice about analysis ECG data


# project Informations
https://incognito-developer.github.io/posts/2023-07-16-aboutEcgPaperAndInfomations



개발이 끝난 프로젝트입니다. 실행은 정상적으로 의도한대로 작동합니다.</br>만약 코드에 문제가 있거나, 필요한 개선 사항이 있다면 [blog](https://incognito-developer.github.io/posts/2023-07-16-aboutEcgPaperAndInfomations) 댓글에 남겨주세요.</br> 확인후 수정을 하겠습니다.</br>
This project has been developed. You can run this code now as intended.</br> If you have any problems or if you need any improvements with code, please leave them in [blog](https://incognito-developer.github.io/posts/2023-07-16-aboutEcgPaperAndInfomations) comments.</br> I'll check and edit them.  

자세한 내용을 알고 싶으시면, `project paper*.pdf`를 읽어주십시오.</br>
If you want to know more detail, please read `project paper*.pdf`

목적: ECG데이터를 학습하고 테스트 하는 것을 위한 프로젝트입니다. </br>
Purpose: training and test model for ECG data.

소스코드는 아직 보기좋게 정리하지 않았습니다. 주석에 달린 링크를 보고 수정하시면 될 것 같습니다.</br>
Source codes are not yet nicely organized. Please edit codes with comments.

Lisence 파일보다 이 글이 우선합니다.</br> 상업적 이용 금지. 이 프로젝트로 인해 발생한 문제점은 책임지지 않습니다.
참고나 수정 시 원본 출처를 명확히 달아주십시오. 
Issues에 이 프로젝트를 참고한 repository를 알려주시면 감사합니다.</br>
This article takes precedence over the license file.</br> No commercial use. We are not responsible for any problems caused by this project.
When making references or modifications, please clearly indicate the original source.
Thank you for letting us know the repository that referenced this project in Issues.

이 프로젝트의 데이터셋은 https://www.kaggle.com/datasets/shayanfazeli/heartbeat 를 사용했습니다.</br>
The dataset for this project is https://www.kaggle.com/datasets/shayanfazeli/heartbeat.

기타 문의사항이나 문제는 [blog](https://incognito-developer.github.io/posts/2023-07-16-aboutEcgPaperAndInfomations)의 댓글에 남겨주세요.</br>
Please leave a message at [blog](https://incognito-developer.github.io/posts/2023-07-16-aboutEcgPaperAndInfomations) comments if you have any questions or problem.

# Usage(사용방법)
1. build your env with ECGanalysis-main/explainForEnvironments/requirements.yaml
2. The codes which in `ECGanalysis-main/code` are final code. These codes are working with whole models, so you can test whole model with `ecgTest.py` and `performanceEvaluation.py`.
3. Directories under `ECGanalysis-main/code/` such as `basic`, `basicWithDropout`... are models directory. You can check models train result, models, layers, and so on.
4. you should edit codes to match your environments. please check and edit these parts. `#to setup enviornments` and `if __name__=="__main__"`

# related project
web version:
https://github.com/incognito-developer/ecgAnalysisUsingML


# related paper
blog:
https://incognito-developer.github.io/posts/2023-07-16-aboutEcgPaperAndInfomations
