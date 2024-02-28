# Day10_Google Colaboratory (Colab)



- ### 딥러닝 개발환경 (standalone)

  - 구축
    - Tensorflow, Keras, PyTorch 등의 딥러닝 라이브러리를 개인 PC에 설치한다.
    - 병렬 처리가 가능한 GPU를 사용하면 개발과 테스트 수행 시 성능을 높일 수 있다.
  - 문제점
    - 호환성 문제로 인해서 라이브러리가 설치되지 않을 수 있다.
    - 딥러닝 라이브러리 간의 종속적인 관계(dependency)를 파악해야 한다.
      - Transfer Learning에 필요한 Tensorflow-hub 라이브러리는 Tensorflow 버전 1.7 이상에서만 설치된다.
    - 개발과 테스트 성능을 높이기 위해서 GPU 같은 고가의 하드웨어를 별도 구매해야 한다.



- ### Google Colaboratory (Colab)

  - 딥러닝 개발을 위한 라이브러리가 이미 설치되어 있다.

  - 고가인 GPU도 저렴하게 또는 무료로 이용할 수 있다.

  - GitHub 등과의 연동을 통해서 자유롭게 소스를 올리고 가져올 수 있는 개발 환경이다.

  - standalone 방식 문제점을 해결한다.

  - #### 개요

    - Gmail 계정이 있는 개발자라면 누구나 무료로 사용할 수 있는 클라우드 서비스이다.
    - 파이썬과 Tensorflow, Keras 등의 딥러닝 라이브러리 등이 미리 설치되어 있기 때문에 웹 브라우저만으로 주피터 노트북(Jupyter Notebook) 작업을 할 수 있으며, 무엇보다도 가장 매력적인 것은 GPU를 무료로 사용할 수 있다는 점이다.
    - Google Colab을 통해서 Google Drive나 GitHub연동을 통해서 자료 공유와 협업을 쉽게 할 수 있는 장점이 있다.

  - #### 접속 및 Jupyter Notebook 실행

    - Google Colab은 다음의 사이트 (https://colab.research.google.com)에 접속 한 후에, Gmail 계정으로 login하면 다음과 같은 시작 페이지를 볼 수 있다.
    - ![image-20231002170753630](Day10_Google Colaboratory (Colab).assets/image-20231002170753630.png)

    - 새 python3 노트를 클릭하면 아래의 그림처럼 나타난다
    - ![image-20231002170906095](Day10_Google Colaboratory (Colab).assets/image-20231002170906095.png)

  - #### 시스템 사양 

    - Colab의 Jupyter Notebook에서 cat, Is, head, wget등의 리눅스 쉘 명령어를 사용하기 위해서는 명령문 앞에 느낌표를 붙이면 Jupyter Notebook에서 바로 실행 할 수 있다.
    - ![image-20231002171205867](Day10_Google Colaboratory (Colab).assets/image-20231002171205867.png)

  - #### 소프트웨어 버전 확인 및 설치

    - ![image-20231002171357570](Day10_Google Colaboratory (Colab).assets/image-20231002171357570.png)

  - #### GPU 사용하기

    - Colab 메뉴의 '런타임' -> '런타임 유형 변경' -> '하드웨어 가속기' 설정을 None에서 GPU로 변경하여 저장하면 GPU를 사용할 수 있다.
    - ![image-20231002171622096](Day10_Google Colaboratory (Colab).assets/image-20231002171622096.png)

  - #### Local PC에 있는 Jupyter Notebook 파일 올리기

    - Colab 메뉴의 '파일' -> '노트 업로드' -> '파일선택' 하면 Local PC에 있는 Jupyter Notebook 파일을 업로드 할 수 있다.
    - ![image-20231002210545560](Day10_Google Colaboratory (Colab).assets/image-20231002210545560.png)

  - #### Local PC에 있는 일반 파일 업로드

    - Colab의 files.upload() 메서드를 통해서 Local PC에 있는 일반 파일을 업로드 할 수 있다.
    - ![image-20231002210732290](Day10_Google Colaboratory (Colab).assets/image-20231002210732290.png)

  - #### Local PC로 파일 다운로드

    - Colab의 files.download() 메서드를 통해서 파일을 다운로드 할 수 있다.
    - ![image-20231002210938530](Day10_Google Colaboratory (Colab).assets/image-20231002210938530.png)



- ### Google Colab - GitHub 연동

  - Cloning 하고자 하는 깃허브 Repository로 이동하여 Clone을 하기 위한 Web URL을 복사한다.
  -  복사된 URL을 Colab으로 가져와서 다음과 같이 git clone 명령어를 실행한다.
  - ![image-20231002221913021](Day10_Google Colaboratory (Colab).assets/image-20231002221913021.png)



- ### Google Colab - Google Drive 연동

  - Colab에 Google Drive를 연결하면 Google Drive에서 파일을 읽거나 저장할 수 있다.
  - Colab에서 아래와 같은 코드를 실행하면 Google Drive를 /content/working_drive/라는 이름으로 Colab에 마운트(mount)시킬수 있다. (working_drive는 임의의 디렉토리 이름이다.)
  - 만약 Colab과 Google Drive를 처음 연결하는 경우라면 아래와 같이 Google Drive 권한 요청을 하는데, 제시된 URL링크를 클릭하면 구글 로그인 과정을 거쳐 최종적으로 비밀키 문자열을 알려준다.
  - ![image-20231002222424931](Day10_Google Colaboratory (Colab).assets/image-20231002222424931.png)

  - 비밀키 문자열을 복사하여 입력칸에 붙여 넣으면 Google Drive와 연결되는 것을 알 수 있다.
  - ![image-20231002222616670](Day10_Google Colaboratory (Colab).assets/image-20231002222616670.png)