# 페달키퍼

[![데모 비디오](http://img.youtube.com/vi/y_dw6AWiR-8/0.jpg)](https://youtube.com/shorts/y_dw6AWiR-8)

이 저장소는 2023 임베디드 경진대회 결선 참여를 위한 코드가 정리된 곳입니다.

데이터의 전처리, 모델 학습 및 로드를 위한 코드가 제공됩니다.

## 라즈베리 파이
### 준비물 (권장)
- **64GB** 이상의 Micro SD Card
- Raspberry Pi 4, **RAM 8GB**
- [페달키퍼 라즈베리파이 이미지](https://drive.google.com/file/d/1sKXNzsizAHgauqctyeqQG7Tr2sQhtBi8/view?usp=drive_link)

### 소개
이 저장소에서는 라즈베리파이의 골치아픈 Dependency 설치 문제를 해결하기 위해, 모든 것이 미리 세팅된 라즈베리파이 SD 카드 이미지 파일을 제공합니다.

Github의 용량 제한으로 구글 드라이브에 별도로 올려 공유드리니, 위의 준비물 항목의 링크 참조 바랍니다.

### 사용법

1. 이미지를 다운로드 후, sd카드 복원 프로그램 ([Etcher](https://etcher.balena.io/))를 통해 img 파일 내 내용물을 SD카드에 복사하십시오.
2. SD카드를 라즈베리파이에 삽입하고, 전원을 인가하여 터미널에 접속하십시오.
3. 미리 설치된 파일 중 `real_time_inference.py`를 실행시키면, 구현된 신경망을 바로 실행할 수 있습니다.

## 신경망 학습
해당 코드를 기반으로 새로운 프로젝트를 시작하고 싶으신 분들을 위해, 커스텀이 가능한 직접 설치 방법 또한 제공해드립니다. 직접 설치 이전에, 다음 작업들이 필요합니다.

### 준비물
- [train_env.yml](https://drive.google.com/file/d/1RNqYarMMfQSzsg_9ZS-AhjV9ssfrHW1d/view?usp=sharing) 파일
- [Anaconda](https://www.anaconda.com/download) 최신 버전
- NVIDIA 그래픽 카드
- 우분투 22.04
### 상세 과정
1. Anaconda 설치.
2. `conda create -n "evt2" --file env.yml`명령어 입력하여 필요 라이브러리 설치.
3. `git clone https://github.com/Dictor/EventTransformer-ESW.git` 명령으로 Repository 클론.
4. `git checkout on-rasp` 명령으로 최신 branch로 변경.
5. `python train.py` 명령을 통해 학습 시작

### 저장소 관리자 연락처
본 저장소의 이용에 어려움이 있거나, 동작되지 않는 링크가 있으면 아래 연락처로 연락 부탁드립니다.

- 김정현 (모델 학습) – kimdictor@gmail.com
- 이주한 (데이터셋 생성 및 변환) – leejoohan9809@gmail.com
- 신현호 (Google Drive Links) – tlsgusghq@kaist.ac.kr

최종 작성자. 신현호
