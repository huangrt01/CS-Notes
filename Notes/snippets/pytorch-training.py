# grad
x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
x.grad


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

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
    self.fc1 = nn.Linear(input_size, 10)
    self.fc2 = nn.Linear(10, 1)
    self.fc3 = nn.Linear(10, 10)

  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    x = torch.sigmoid(x) * 100
    return x

def train_deep_learning_model(dataset_file):
  data, labels = asyncio.run(
      read_data_from_file(dataset_file,
                          FEATURE_LIST,
                          select_indices=SELECT_INDICES))

  scaler = MinMaxScaler()
  scaler.fit(data)

  joblib.dump(scaler, SCALER_FILENAME)
  data = scaler.transform(data)

  # set random_state
  X_train, X_val, y_train, y_val = train_test_split(data,
                                                    labels,
                                                    test_size=0.05,
                                                    random_state=42)

  train_dataset = RankDataset(X_train, y_train)
  train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)

  val_dataset = RankDataset(X_val, y_val)
  val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)

  model = RankModel(len(FEATURE_LIST))
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01)
  scheduler = ReduceLROnPlateau(optimizer,
                                mode='min',
                                factor=0.7,
                                patience=80,
                                min_lr=0.001)

  for epoch in range(1000):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
      inputs = torch.tensor(inputs, dtype=torch.float32)
      targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(
          1)  # Add this line to reshape the targets

      optimizer.zero_grad() # 需要zero_grad，否则会累加
      outputs = model(inputs)
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
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(
            1)  # Add this line to reshape the targets
        outputs = model(inputs)
        val_loss += criterion(outputs, targets).item()
      val_epoch_loss = val_loss / len(val_loader)
      scheduler.step(val_epoch_loss)
    logging.info(
        f'Epoch {epoch+1}/{500}, Loss: {epoch_loss:.4f}, Eval Loss: {val_epoch_loss:.4f}, lr: {current_lr:.6f}'
    )

  torch.save(model.state_dict(),
             os.path.join(os.path.dirname(__file__), 'deep_learning_model.pth'))