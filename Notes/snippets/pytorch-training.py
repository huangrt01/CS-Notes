### grad
x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
x.grad

torch.no_grad()禁用梯度

with torch.no_grad():
	y = x * 2

@torch.no_grad()
def doubler(x):
	return x * 2

with torch.no_grad():
	a = torch.nn.Parameter(torch.rand(10))
	a.requires_grad

# 例外：
所有工厂函数或创建新张量的函数，都不受此模式的影响
工厂函数是用于生成tensor的函数。常见的工厂函数有torch.rand、torch.randint、torch.randn、torch.eye等
pytorch觉得你是创建网络的训练权重，所以此时requires_grad的默认值是true。

# requires_grad默认是False

### load&store


x = torch.tensor([0, 1, 2, 3, 4])
torch.save(x, "tensor.pt")

torch.load("tensors.pt", weights_only=True)
torch.load("tensors.pt", map_location=torch.device("cpu"), weights_only=True)
state = torch.load("tensors.pt", map_location=torch.device("cuda:0"), weights_only=True)

model = MyNet()

torch.save(model.state_dict(), "model_weights.pth")
model.load_state_dict(state)

# OrderedDict([
#             ("nn1.weight", tensor([[-0.1838, -0.2477],[ 0.4845,  0.3157],[-0.5628,  0.3612]])), 
#             ("nn1.bias", tensor([-0.4328, -0.6779,  0.3845])), 
#             ("nn2.weight", tensor([[-5.0585e-01, -4.6973e-01,  1.6044e-02],[-3.4606e-01,  1.1130e-01, -2.0727e-01],
#                                     [-3.9844e-02, -4.2531e-01,  8.2558e-02],[ 3.3171e-02, -3.4334e-01,  4.5039e-01],
#                                     [-2.5320e-04, -5.2037e-01,  1.3504e-02],[-3.0776e-01,  8.9345e-02, -1.1076e-01]])),                 
#             ("nn2.bias", tensor([ 0.1229, -0.2344,  0.0568, -0.3430,  0.2715, -0.3521]))
# ])

### parameter

torch.nn.parameter.Parameter(data=None, requires_grad=True)
torch.nn.parameter.UninitializedParameter(requires_grad=True, device=None, dtype=None)

# UninitializedParameter多用于lazy模块
- 根据shape、device，lazy initialize

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.weight * x + self.bias

class CustomModule(nn.Module):
    def __init__(self):
        super(CustomModule, self).__init__()
        self.weight = torch.nn.parameter.UninitializedParameter(requires_grad=True)
        self.bias = torch.nn.parameter.UninitializedParameter(requires_grad=True)

    def forward(self, x):
        if self.weight.shape[0]!= x.shape[1]:
            self.weight.data = torch.randn(x.shape[1], requires_grad=True)
            self.bias.data = torch.randn(1, requires_grad=True)
        output = torch.matmul(self.weight, x) + self.bias
        return output


### 动态图
loss1 = z.mean()
loss2 = z.sum()
print(loss1,loss2)
loss1.backward(retain_graph=True)
loss2.backward()



### Example

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

async def process_file(src_file, output_file):
  with open(src_file, 'r') as src_stream:
    data = json.load(src_stream)
  with open(output_file, 'w') as dst_stream:
    for i, item in enumerate(data):
      if i % 100 == 0:
        logging.info(f"process item {i}")
      try:
        item = await process_item(item)
        json.dump(item, dst_stream, ensure_ascii=False)
        dst_stream.write('\n')
      except Exception as e:
        logging.error(f"process item failed: {e}, {item}")
        continue
  return data


async def read_data_from_file(file,
                              feature_list: list[str],
                              select_indices=None):
  assert isinstance(feature_list, list)
  assert all(isinstance(f, str) for f in feature_list)
  assert select_indices is None or isinstance(select_indices, list)
  with open(file, 'r') as stream:
    data = [json.loads(line) for line in stream.readlines()]
    labels = []
    features = []
    for i, item in enumerate(data):
      try:
        labels.append(item["avg"] / 100)
        feature = [item[f] for f in feature_list]
        features.append(feature)
        if i % 100 == 0:
          logging.info(f"read data from file, item num: {i}")
      except Exception as e:
        logging.error(f"process data failed: {e}")
        continue
  features_df = pd.DataFrame(features, columns=feature_list)
  correlation_matrix = features_df.corr()

  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', None)
  pd.set_option('display.max_rows', None)
  print("correlation_matrix:", correlation_matrix)
  pd.reset_option('display.max_columns')
  pd.reset_option('display.width')
  pd.reset_option('display.max_rows')

  labels = np.array(labels)
  features = np.array(features)
  if select_indices:
    features = np.take(features, select_indices, axis=-1)
  return features, labels


class RankDataset(Dataset):

  def __init__(self, data, labels):
    self.data = data
    self.labels = labels

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]


class RankModel(nn.Module):

  def __init__(self, input_size):
    super(RankModel, self).__init__()
    self.fc1 = nn.Linear(input_size, 50)
    self.fc2 = nn.Linear(50, 1)
    self.fc3 = nn.Linear(50, 50)

    # self.fc1 = nn.Linear(input_size, 10)
    # self.fc2 = nn.Linear(10, 1)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc3(x))
    x = self.fc2(x)
    x = torch.sigmoid(x)

    # x = torch.relu(self.fc1(x))
    # x = self.fc2(x)
    # x = torch.sigmoid(x)
    return x.squeeze()

  def predict(self, data):
    self.eval()
    with torch.no_grad():
      data = torch.tensor(data, dtype=torch.float32)
      predictions = self(data)
    return predictions.numpy()


@lru_cache
def get_feature_list(model_name: str):
  feature_list = FEATURE_LIST_MAP[model_name]
  return feature_list


@lru_cache
def get_rank_model(model_name: str):
  model = RankModel(len(FEATURE_LIST_MAP[model_name]))
  model.load_state_dict(
      torch.load(os.path.join(os.path.dirname(__file__), f'{model_name}.pth'),
                 weights_only=True))
  return model


def rank_model_predict(datas, model_name):
  scaler = joblib.load(
      os.path.join(os.path.dirname(__file__), f'{model_name}.scaler'))
  datas = [[item[f] for f in get_feature_list(model_name)] for item in datas]
  datas = np.array(datas)
  datas = scaler.transform(datas)
  if SELECT_INDICES is not None:
    datas = np.take(datas, SELECT_INDICES, axis=-1)
  model = get_rank_model(model_name)
  prediction = model.predict(datas).tolist()
  return prediction


def train_deep_learning_model(dataset_file, model_name):
  data, labels = asyncio.run(
      read_data_from_file(dataset_file,
                          FEATURE_LIST_MAP[MODEL_NAME],
                          select_indices=SELECT_INDICES))

  scaler = MinMaxScaler()
  scaler.fit(data)

  joblib.dump(scaler,
              os.path.join(os.path.dirname(__file__), f'{model_name}.scaler'))
  data = scaler.transform(data)

  X_train, X_val, y_train, y_val = train_test_split(data,
                                                    labels,
                                                    test_size=0.05,
                                                    random_state=42)

  train_dataset = RankDataset(X_train, y_train)
  train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)

  val_dataset = RankDataset(X_val, y_val)
  val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)

  model = RankModel(len(FEATURE_LIST_MAP[MODEL_NAME]))
  criterion = nn.SmoothL1Loss(reduction='mean', beta=0.1)
  # criterion = nn.L1Loss()
  # criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01)  # weight_decay=0.001
  scheduler = ReduceLROnPlateau(optimizer,
                                mode='min',
                                factor=0.6,
                                patience=60,
                                min_lr=0.001)
  best_epoch = 0
  best_loss = float('inf')
  no_improvement_count = 0

  for epoch in range(1000):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
      inputs = torch.tensor(inputs, dtype=torch.float32)
      targets = torch.tensor(targets, dtype=torch.float32)

      optimizer.zero_grad()
      outputs = model(inputs)
      # logging.info(f'ruiteng {outputs} {targets}')
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
      for inputs, targets in val_loader:
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(
            targets,
            dtype=torch.float32)  # Add this line to reshape the targets
        outputs = model(inputs)
        val_loss += criterion(outputs, targets).item()
      val_epoch_loss = val_loss / len(val_loader)
      scheduler.step(val_epoch_loss)
    logging.info(
        f'Epoch {epoch+1}/{500}, Loss: {epoch_loss:.4f}, Eval Loss: {val_epoch_loss:.4f}, lr: {current_lr:.6f}'
    )

    if val_epoch_loss < best_loss:
      best_loss = val_epoch_loss
      best_epoch = epoch
      no_improvement_count = 0
      torch.save(
          model.state_dict(),
          os.path.join(os.path.dirname(__file__),
                       f'rank_model_{epoch}.pth'))
    else:
      no_improvement_count += 1
      if no_improvement_count >= 200:
        logging.info(f'Early stopping at epoch {epoch+1}')
        break

  logging.info(f'Best Epoch: {best_epoch+1}, Best Eval Loss: {best_loss:.4f}')