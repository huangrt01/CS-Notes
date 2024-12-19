import torch
from torchvision import transforms
from torch.utils.data import Dataset





class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = [...] # 根据需求加载数据集

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 加载数据
        img_path = self.data[index]["image_path"]
        label_path = self.data[index]["label_path"]
        img = self.load_image(img_path)
        label = self.load_label(label_path)

        # 数据预处理
        if self.transform:
            img = self.transform(img)

        return img, label

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in dataloader:
    ...


X = torch.randn(100, 3)
Y = torch.randn(100, 1)

dataset = TensorDataset(X, Y)