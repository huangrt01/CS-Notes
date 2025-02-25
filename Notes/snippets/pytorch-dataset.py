import torch
from torchvision import transforms
from torch.utils.data import Dataset

torch.utils.data.Dataset需要覆写下面两个方法

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


### 内存问题 -- 读多个文件的实现

* 读数据
  * https://zhuanlan.zhihu.com/p/376974245
  * Dataset 每次获取一个Part的Dataframe，外部再进行batch_size的划分，这样在整个迭代期间，最多只会有num_worker个Dataset被实例化，事实上也确实不再有内存溢出的问题

class ExpDataset2(Dataset):
    def __init__(self, filenames, features_config): 
        self._filenames = filenames
        
    def __getitem__(self, idx):
        path = self._filenames[idx]
        return preprocess(read_csv(path)

def load_data(paths, features_config, num_workers, batch_size):
    dataset = ExpDataset2(paths, features_config)
    data = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=1,
        collate_fn=collate_fn2
    )
    for df in data:
        for idx_from in range(0, df.shape[0], batch_size):
            yield examples[idx_from : idx_from + batch_size]


DistributedSampler 要求输入的数据集是可索引的（map-style dataset）

to_map_style_dataset 函数的作用是将迭代式数据集转换为可索引的数据集，使得数据集可以被 DistributedSampler 使用。
通过这种转换，我们可以为数据集添加 __getitem__ 和 __len__ 方法，从而满足 DistributedSampler 的要求。


### IterableDataset

def worker_init_fn(worker_id):
...     worker_info = torch.utils.data.get_worker_info()
...     dataset = worker_info.dataset  # the dataset copy in this worker process
...     overall_start = dataset.start
...     overall_end = dataset.end
...     # configure the dataset to only process the split workload
...     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
...     worker_id = worker_info.id
...     dataset.start = overall_start + worker_id * per_worker
...     dataset.end = min(dataset.start + per_worker, overall_end)

print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))



### dataset加速 -- prefetcher

最早见于nvidia apex

https://blog.csdn.net/weiman1/article/details/125610786
