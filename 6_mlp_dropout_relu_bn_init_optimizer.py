"""
### Step 1. 모듈 불러오기
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""### Step 2. 딥러닝 모델 설계 시 필요한 장비 세팅 """

if torch.cuda.is_available():
  DEVICE = torch.device('cuda')
else:
  DEVICE = torch.device('cpu')

print("PyTorch Version:", torch.__version__, ' Device:', DEVICE)

BATCH_SIZE = 32 # 데이터가 32개로 구성되어 있음. 
EPOCHS = 10 # 전체 데이터 셋을 10번 반복해 학습함.

"""### Step 3. 데이터 다운로드
- torchvision 내 datasets 함수 이용하여 데이터셋 다운로드 합니다.
- ToTensor() 활용하여 데이터셋을 tensor 형태로 변환
- 한 픽셀은 0~255 범위의 스칼라 값으로 구성, 이를 0~1 범위에서 정규화 과정 진행
- DataLoader는 일종의 Batch Size 만큼 묶음으로 묶어준다는 의미
  + Batch_size는 Mini-batch 1개 단위를 구성하는 데이터의 개수
  
"""

train_dataset = datasets.MNIST(root = "../data/MNIST", 
                               train = True, 
                               download = True, 
                               transform = transforms.ToTensor())

test_dataset = datasets.MNIST(root = "../data/MNIST", 
                              train = False, 
                              transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                           batch_size = BATCH_SIZE, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset, 
                                           batch_size = BATCH_SIZE, 
                                           shuffle=False)

"""### step 4. 데이터 확인 및 시각화
- 데이터를 확인하고 시각화를 진행합니다. 
- 32개의 이미지 데이터에 label 값이 각 1개씩 존재하기 때문에 32개의 값을 갖고 있음
"""

for (X_train, y_train) in train_loader:
  print('X_train:', X_train.size(), 'type:', X_train.type())
  print('y_train:', y_train.size(), 'type:', y_train.type())
  break

pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))
for i in range(10):
  plt.subplot(1, 10, i + 1)
  plt.axis('off')
  plt.imshow(X_train[i, :, :, :].numpy().reshape(28, 28), cmap="gray_r")
  plt.title('Class: ' + str(y_train[i].item()))
plt.show()

"""### step 5. MLP 모델 설계
- torch 모듈을 이용해 MLP를 설계합니다.

#### Dropout
- Dropout은 신경망이 지니고 있는 단점인 과적합과 Gradient Vanishing을 완화시킬 수 있는 여러 알고리즘 중 하나
- Layer의 노드를 랜덤하게 Drop하면서 Generalization 효과를 가져오게 하는 테크닉
  + 논문: [Dropout: A Simple Way to Prevent Neural Networks from
Overfitting](https://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)
- Dropout 적용한다는 것은 Weight Matrix에 랜덤하게 일부 Column에 0을 추가하여 연산하는 것과 비슷함. 
  + 즉, 과적합 방지용으로 사용됨. 
- 수정된 코드 설명
  + (1): 몇 퍼센트의 노드에 대해 가중값을 계산하지 않을 것인지를 명시해주는 부분
  + (2): Sigmoid( ) 함수의 결괏값에 대해 Dropout을 적용하는 부분. 계산되는 과정 속에 있는 x 값에 적용하며 p값은 몇 퍼센트의 노드에 대해 계산하지 않을 것인지를 조정하는 요소. `training = self.training` 학습 시, 검증 상태에 따라 다르게 적용되기 위해 존재하는 파라미터.

#### Relu
- Activation 함수는 어떤 신호를 입력받아 이를 적절히 처리해 출력해주는 함수를 의미
  + Default: 주로 시그모이드 함수를 사용함. 
- Back Propagation 과정 중에 시그모이드를 미분한 값을 계속 곱해주면 Gradient 값이 앞 단의 Layer로 올수록 0으로 수렴하는 현상이 발생함. 
  + Gradient Vanishing, Hidden Layer가 깊어질수록 이러한 현상이 짙어짐
- 시그모이드 함수 처럼, 비선형 활성 함수가 지니고 있는 문제점을 어느 정도 해결한 활성 함수. 
- Relu = f(x) = max(0, x)와 같이 정의됨. 
  + 입력값이 x > 0: 그대로 출력
  + 입력값이 x < 0: 0으로 출력

#### Batch Normalization
- Internal Covariance Shift 현상이 발생함. 
  + 각 Layer마다 Input 분포가 달라짐에 따라 학습 속도가 느려지는 현상을 말함. 
  + 효과: Layer의 Input 분포를 정규화해 학습 속도를 빠르게 하겠다는 것을 의미함. 또한, Gradient Vanishing 문제도 완화해줌. 
- 추가 설명: https://wegonnamakeit.tistory.com/47
- Batch Normaliation은 1-Dimension, 2-Dimension, 3-Dimension 등 다양한 차원에 따라 적용되는 함수명이 다르기 때문에 유의해서 사용해야 한다. MLP내 각 Layer에서 데이터는 1-Dimension 크기의 벡터 값을 계산하기 때문에 nn.BatchNorm1d()을 이용한다.
"""

class Net(nn.Module):
  '''
  Forward Propagation 정의
  '''
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(28 * 28 * 1, 512) # (가로 픽셀 * 세로 픽셀 * 채널 수) 크기의 노드 수 설정 Fully Connected Layer 노드 수 512개 설정
    self.fc2 = nn.Linear(512, 256) # Input으로 사용할 노드 수는 512으로, Output 노드수는 256개로 지정
    self.fc3 = nn.Linear(256, 10) # Input 노드수는 256, Output 노드수는 10개로 지정
    self.dropout_prob = 0.3 
    self.batch_norm1 = nn.BatchNorm1d(512) # (1) 첫번재 fc1의 Output이 512이기 때문
    self.batch_norm2 = nn.BatchNorm1d(256) # (2) 두번째 fc2의 Output이 256이기 때문


  def forward(self, x):
    x = x.view(-1, 28 * 28) # 1차원으로 펼친 이미지 데이터 통과
    x = self.fc1(x)
    x = self.batch_norm1(x) # (3) 첫번째 Fully
    x = F.relu(x) # F.sigmoid(x)
    x = F.dropout(x, training = self.training, p = self.dropout_prob) # (2)
    x = self.fc2(x)
    x = self.batch_norm2(x) # (4)
    x = F.relu(x) # F.sigmoid(x)
    x = F.dropout(x, training = self.training, p = self.dropout_prob) # (3)
    x = self.fc3(x)
    x = F.log_softmax(x, dim = 1)
    return x

"""### step 6. Initialization 설정
- Back Propagation 설정 위한 목적 함수 설정 
- Initialize는 초기화. 신경망은 처음에 Weight를 랜덤하게 초기화하고 Loss가 최소화되는 부분을 찾아감. 
- 초기 분포로 Uniform Distribution이나 Normal Distribution 사용. 
- 그러나, 초깃값에 따라, 학습속도가 달라지는 현상이 생김. 
- Xavier Initialization, LeCun Initialization, He Initialization 등이 있음.

### step 7. Optimizer
- Batch 단위로 Back Propagation 하는 과정을 Stochastic Gradient Descent(SGD)라 하고 이러한 과정을 `Optimization`이라고 함. 
- 종류는 다음과 같음. 
  + Momentum
  + NAG
  + Adagrad
  + RMSProp
  + Adadelta
  + Adam
  + RAdam

- 각 옵티마이저중 절대적으로 우세한 것은 없습니다. 따라서, 여러 가지 방식으로 시도해보는 것이 중요하다.
"""

import torch.nn.init as init
def weight_init(m): # (1) 
  if isinstance(m, nn.Linear): # (2)
    init.kaiming_uniform_(m.weight.data) # (3)

model = Net().to(DEVICE)
model.apply(weight_init)
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.CrossEntropyLoss() # output 값과 원-핫 인코딩 값과의 Loss 
print(model)

"""- (1): Feature 값으로 이용되는 데이터를 설계한 모델의 Input으로 사용해 Output을 계산
- (2): 계산된 Output을 Input으로 이용한 Feature 값과 매칭되는 레이블 값을 기존에 정의한 Objective function을 통해 Loss값으로 계산
- (3): 계산된 Loss값을 통해 Gradient를 계산해 모델 내 파라미터 값을 Back Propagation에 의해 업데이트.

### step 8. MLP 모델 학습
- MLP 모델을 학습 상태로 지정하는 코드를 구현
"""

def train(model, train_loader, optimizer, log_interval):
  model.train()
  for batch_idx, (image, label) in enumerate(train_loader): # 모형 학습
    image = image.to(DEVICE)
    label = label.to(DEVICE)
    optimizer.zero_grad() # Optimizer의 Gradient 초기화
    output = model(image)
    loss = criterion(output, label)
    loss.backward() # back propagation 계산
    optimizer.step()

    if batch_idx % log_interval == 0:
      print("Train Epoch: {} [{}/{}({:.0f}%)]\tTrain Loass: {:.6f}".format(Epoch, batch_idx * len(image), 
                                                                           len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

"""### step 9. 검증 데이터 확인 함수

"""

def evaluate(model, test_loader):
  model.eval()
  test_loss = 0
  correct = 0

  with torch.no_grad():
    for image, label in test_loader:
      image = image.to(DEVICE)
      label = label.to(DEVICE)
      output = model(image)
      test_loss += criterion(output, label).item()
      prediction = output.max(1, keepdim = True)[1]
      correct += prediction.eq(label.view_as(prediction)).sum().item()

  test_loss /= len(test_loader.dataset)
  test_accuracy = 100. * correct / len(test_loader.dataset)
  return test_loss, test_accuracy

"""- 모델 평가 시, Gradient를 통해 파라미터 값이 업데이트되는 현상 방지 위해 torch.no_grad() Gradient의 흐름 제어

### step 10. MLP 학습 실행
"""

for Epoch in range(1, EPOCHS + 1):
  train(model, train_loader, optimizer, log_interval=200)
  test_loss, test_accuracy = evaluate(model, test_loader)
  print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n".format(Epoch, test_loss, test_accuracy))

"""- train 함수 실행하면, model은 기존에 정의한 MLP 모델, train_loader는 학습 데이터, optimizer는 SGD, log_interval은 학습이 진행되면서 mini-batch index를 이용해 과정을 모니터링할 수 있도록 출력함.
- 학습 완료 시, Test Accuracy는 97% 수준의 정확도를 나타냄. 
- sigmoid() 함수 적용할 때 보다 ReLU()함수 적용 시 보다 높은 성능 유지함. 
"""