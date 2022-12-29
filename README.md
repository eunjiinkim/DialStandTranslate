# Korean Dialect <-> Standard Translator

한국어 방언 <-> 표준어 번역기를 훈련하는 코드입니다.


* 모델과 코드는 SKT의 [KoBART](https://github.com/SKT-AI/KoBART)를 활용하였습니다.

* 다음은 표준어를 제주어로 번역한 예시입니다.

<img width='70%' src="https://user-images.githubusercontent.com/55074165/209938167-3b22ab9c-0c89-493f-80f0-832124545a4a.png">

## Data
* AI hub의 [표준어-방언 병렬 데이터](https://aihub.or.kr/aihubdata/data/list.do?pageIndex=1&currMenu=115&topMenu=100&dataSetSn=&srchdataClCode=DATACL001&srchOrder=&SrchdataClCode=DATACL002&searchKeyword=%EB%B0%A9%EC%96%B8)를 활용하였습니다.
1. json 데이터를 tsv 파일로 처리하여 아래와 같이 저장합니다.
  
  * 방언과 표준어가 다른 경우만 추출하고, 최소쌍에 해당하는 부분을 함께 저장합니다.
  * extract_data.ipynb
  <img width="70%" alt="스크린샷 2022-12-29 오후 7 47 04" src="https://user-images.githubusercontent.com/55074165/209940596-f09a2942-f661-4363-9161-a2b9ef77140e.png">
  * data/{region}/{train/test}/{train/test}_data.tsv 에 저장됩니다.
  
2. 전처리한 파일을 최종적으로 data/{region}/train_cleaned.tsv와 같이 저장하였습니다.

## How to train
### dialect > standard

````
python3 trainer_d2s.py --region 'jeju'
````

### standard < dialect

````
python3 trainer_s2d.py --region 'jeju'
````
