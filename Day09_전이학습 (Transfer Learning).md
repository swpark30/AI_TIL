# Day09_전이학습 (Transfer Learning)



- ### MNIST 인식을 위한 CNN 아키텍처와 소요시간

  - MNIST (0~9를 나타내는 흑백 이미지)를 높은 정확도로 인식하기 위해서는, 최소 3개의 컨볼루션 층과 1개의 완전연결 층이 필요하며, 이때 전체 학습에 소요되는 시간은 1개의 CPU 환경에서 약 1시간 10분 소요된다.



- ### 고해상도 칼라 이미지 인식을 위한 CNN 아키텍처와 소요시간

  - 아키텍처
    - CIFAR-10 / ImageNet에서 사용되는 고해상도 칼라 이미지를 인식하기 위해서는, 최소 5개 이상의 컨볼루션 층과 2개 이상의 완전연결 층을 이용하여 입력으로 주어지는 복잡한 이미지의 특징을 추출하는 학습 과정을 거쳐야 한다.
  - ![image-20231002141134978](Day09_전이학습 (Transfer Learning).assets/image-20231002141134978.png)
  - 소요시간
    - 학습에 소요되는 시간은 1개의 CPU 환경이라면 수백 ~ 수천시간이 소요될 수도 있다.

  - #### 고해상도 칼라 이미지에 잘 훈련된 사전학습(pre-trained)된 CNN 모델이 있다면, 

    - 이러한 CNN 모델을 바탕으로 우리가 분석하고자 하는 이미지 데이터에 맞도록, 이미 학습되어 있는 CNN 모델의 다양한 파라미터 등을 수정해서 사용한다.
    - 임의의 값으로 초기화된 파라미터를 처음부터 학습시키는 것에 비해 소요시간을 획기적으로 줄일 수 있으며 다양한 이미지 데이터를 짧은 시간에 학습 할 수 있는 장점이 있다. 



- ### Transfer Learning (전이학습)

  - ![image-20231002142228896](Day09_전이학습 (Transfer Learning).assets/image-20231002142228896.png)
  - 실무에서는 고해상도 칼라 이미지를 학습하는 경우, CNN 아키텍처를 구축하고 임의의 값으로 초기화된 파라미터 값(가중치, 바이어스 등) 들을 처음부터 학습시키지 않고 대신 고해상도 칼라 이미지에 이미 학습되어 있는 모델의 가중치와 바이어스를 자신의 데이터로 전달하여 빠르게 학습하는 방법이 일반적이다.
  - 이처럼 고해상도 칼라 이미지 특성을 파악하는데 있어 최고의 성능을 나타내는 ResNet, GoogLeNet을 이용하여 우리가 원하는 데이터에 미세조정 즉, **Fine Tuning**으로 불리는 작은 변화만을 주어 학습시키는 방법을 Transfer Learning이라고 한다.



- ###  Trained Model (Google Inception Model, MS ResNet Model ...)

  - ![image-20231002142518571](Day09_전이학습 (Transfer Learning).assets/image-20231002142518571.png)



- ### Google Inception-V3를 이용한 실습

  - ![image-20231002144437306](Day09_전이학습 (Transfer Learning).assets/image-20231002144437306.png)

  - #### 실습 순서
  
    1. Tensorflow 설치
       - pip install tensorflow
    
    2. Tensorflow Hub 설치
       - pip install tensorflow-hub
    
    3. retrain.py 파일 다운로드
    
       - 고해상도 칼라이미지의 기본특징들이 이미 학습되어 있는 Google Inception-v3 소스 retain.py
    
       - https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py
    
    4. flower_photos.tgz 파일 다운로드 및 압축해제
    
       - Transfer Learning 실습을 위하여 Google에서 기본적으로 제공하는 꽃 분류 Training Data
    
       - https://download.tensorflow.org/models/example_images/flower_photos.tgz
    
    5. label_image.py 파일 다운로드
    
       - Training Data로 학습을 마친 후, 임의의 이미지를 분류하고 정확도를 확인하는 label_image.py
    
       - https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/label_image/label_image.py
    
       - 다운로드 사이트가 변경될 시 https://github.com/neowizard2018/neowizard/tree/master/MachineLearning 에서도 다운 가능하다.
    
    6. Training Data(flower_photos)를 이용하여 파인튜닝 수행
    
       - 다음과 같이 `python ./retrain.py --image_dir=./flower_photos`를 이용하여 Training Data를 재학습시키면, 파인튜닝으로 학습된 가중치와 바이어스등의 학습결과는 /tmp에 저장된다.
    
    7. 학습된 내용을 바탕으로 이미지 분류를 수행해주는 label_image.py를 다음과 같이 실행
    
       - `python ./label_image.py --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --input_layper=Placeholder --output_layer=final_result --image=./sunflower5.jpg`



- ### 나만의 데이터를 이용한 Transfer Learning 실습

  - 나만의 이미지로 Training DataSet을 만들기
    1. 먼저 이미지의 정답을 나타내는 이름으로 폴더를 각각 만들어 주어야 한다.
    2. 각 폴더(dog, cat, koala, squirrel 이름의 폴더)안에 직접 찍은 사진이나 인터넷에서 무작위로 다운받은 파일을 최소 50장 이상 저장한다.
    3. 이러한 폴더의 루트 폴더를 animal_photos 등의 임의의 이름으로 지정하고, retrain.py를 실행시킬때 --image_dir 파라미터로 루트폴더 이름을 전달하면 Transfer Learning을 실행할 수 있다.

  - #### 실습 순서

    1. Google Inception에서 했던 순서 1~5 번 중 4번만 빼고 똑같다.

    2. Training Data를 이용하여 파인튜닝 수행

       - 다음과 같이 `python ./retrain.py --image_dir=./animal_photos`를 이용하여 Training Data를 재학습시키면, 파인튜닝으로 학습된 가중치와 바이어스등의 학습결과는 /tmp에 저장된다.

    3. 학습된 내용을 바탕으로 이미지 분류를 수행해주는 label_image.py를 다음과 같이 실행

       - `python ./label_image.py --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --input_layper=Placeholder --output_layer=final_result --image=./test1.jpg`

       