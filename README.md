```
conda activate mediapipe
```

```
$ pip install mediapipe opencv-python 

# hugginh face 
pip install datasets
```

# Mediapipe 
- github : https://github.com/google-ai-edge/mediapipe
MediaPipe는 Google에서 개발한 오픈 소스 플랫폼 프레임워크로, 개발자들이 머신러닝을 기반으로 한 모바일 및 웹 애플리케이션에서 사용할 수 있는 다양한 미리 만들아진 솔루션을 제공한다.

특징은 다음과 같다.

사용하기 편리하다.
매우 빠르다.

MediaPipe는 GPU 가속을 통해 빠른 처리 성능을 제공한다.
많은 MediaPipe 솔루션은 실시간 애플리케이션에서 사용하기 위해 최적화되어 있다.


커스터마이징이 가능하다.

개발자는 MediaPipe 그래프를 사용하여 자신의 파이프라인을 쉽게 구성하거나 수정할 수 있다.


솔루션 기반이다.

MediaPipe는 얼굴 인식, 손 추적, 포즈 추정 등과 같은 여러 미리 만들어진 ML 솔루션을 제공한다. 이를 통해 개발자들은 복잡한 ML 파이프라인을 간단히 구축하고 사용할 수 있다.
MediaPipe는 오픈 소스 프레임워크로, 개발자들은 코드를 자유롭게 사용, 수정 및 확장할 수 있다.

## Mediapipe hands 
- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index?hl=ko#models
- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python?hl=ko

![캡쳐3](/refer/2024-07-26%20165619.png)

### TFLite 에서 .task로 변경하는 방법 
1. TensorFlow 모델을 TensorFlow Lite 모델로 변환하기
먼저, TensorFlow 모델을 TensorFlow Lite 형식으로 변환합니다.

```
import tensorflow as tf

# TensorFlow 모델 로드 또는 생성
model = create_blazepalm_model()

# 모델 저장 (임시 경로)
model.save('saved_model')

# TensorFlow Lite 모델로 변환
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
tflite_model = converter.convert()

# TensorFlow Lite 모델 저장
with open('blazepalm_model.tflite', 'wb') as f:
    f.write(tflite_model)

```
2. TensorFlow Lite 모델을 TensorFlow Task Library 형식으로 변환하기
TensorFlow Lite 모델을 .task 파일로 변환하려면 TensorFlow Task Library를 사용해야 합니다. 이를 위해 Python API를 사용하여 모델을 변환합니다.

```
from tensorflow_lite_support.metadata.python import metadata_writer_for_object_detector
from tensorflow_lite_support.metadata.python.metadata_writers import object_detector
from tensorflow_lite_support.metadata.python.metadata_writers import writer_utils

# TensorFlow Lite 모델 경로
model_path = 'blazepalm_model.tflite'
# .task 파일 경로
task_file_path = 'blazepalm_model.task'

# 모델 메타데이터 생성
writer = metadata_writer_for_object_detector.MetadataWriter.create_for_inference(
    model_path,
    input_norm_mean=[127.5],
    input_norm_std=[127.5],
    label_file_paths=['path/to/labelmap.txt']
)
# 메타데이터 적용 및 저장
writer_utils.save_file(writer.populate(), task_file_path)

```
위 코드는 TensorFlow Lite 모델에 메타데이터를 추가하고 이를 .task 파일로 저장하는 과정을 보여줍니다.

주요 단계 설명 

    1. TensorFlow 모델을 TensorFlow Lite로 변환:
    - TensorFlow 모델을 저장한 후 tf.lite.TFLiteConverter를 사용하여 TensorFlow Lite 모델로 변환합니다.   

    2. TensorFlow Lite 모델에 메타데이터 추가:
    - metadata_writer_for_object_detector를 사용하여 객체 검출 모델에 필요한 메타데이터를 추가합니다.   

    3. .task 파일로 저장:
    - writer_utils.save_file을 사용하여 최종 .task 파일로 저장합니다.
    이 과정에서는 객체 검출 모델에 맞는 메타데이터를 추가하기 위해 metadata_writer_for_object_detector.MetadataWriter를 사용했습니다. 다른 작업(예: 이미지 분류)에는 해당 작업에 맞는 메타데이터 작성기를 사용해야 합니다.
   
    .task 파일을 생성한 후에는 TensorFlow Task Library를 사용하여 모바일 또는 임베디드 장치에서 모델을 실행할 수 있습니다.

# Datasets 
사용된 데이터 
- https://github.com/twerdster/HandNet

추가 할 수 있는 데이터
- https://huggingface.co/datasets/Vincent-luo/hagrid-mediapipe-hands

# Paper 
- file:///M:/MediaPipe.Hands._.On-device.Real-time.Hand.Tracking.pdf

Face 검출과는 다르게, 손은 고대피 패턴(눈,코,입)이 없기에 신뢰성 있게 감지하기 어렵다. 
따라서 손바닥을 먼저 검출하여 경계상자를 도출하고, 검출 된 경계살자에서 랜드마크의 위치 정확도를 위해 네트워크의 
대부분의 용량을 사용한다. 손바닥 검출기는 새로운 이미지(or frame)에서 진행 되고 Frame 같은 경우 이전 frame에서 경계 상자를 도출하여 사용하고, 손 예측이 손을 놓쳤다고 판단될 때만 다시 손바닥을 검출한다. 

## Palm Detection

1. 손바닥(palm) 검출기 : 손을 감지하는 것 보다 손바닥 or 주먹을 bbox를 예측하는 것이 훨씬 간단하다. 
2. 작은 물체에서도 더 큰 장면 맥락 인식을 위해 FPN과 유사한 인코더-디코더 특징 추출기를 사용한다.    

![캡처](/refer/2024-07-26%20143618.png)  

## Hand Landmark Model
![캡처2](/refer/2024-07-26%20144750.png)


# local 
C:\Users\sim2real\AppData\Local\miniconda3\envs\mediapipe\Lib\site-packages\mediapipe
C:\Users\sim2real\AppData\Local\miniconda3\envs\mediapipe\Lib\site-packages\mediapipe\python\solutions\hands.py


# Mediapipe 추가 정보 
- pose 관련 
    - https://giveme-happyending.tistory.com/202

- example 카메라 관련, BUILD 관련 튜토리얼? 
    - https://medium.com/@mahakal001/end-to-end-project-example-in-mediapipe-b74a4a8ebb61

- Mediapipe 추가 훈련 관련 질문 
    - https://github.com/google-ai-edge/mediapipe/issues/3202
    - https://github.com/google-ai-edge/mediapipe/issues/3410
    - https://github.com/google-ai-edge/mediapipe/issues/3727
    - https://github.com/google-ai-edge/mediapipe/issues/507

