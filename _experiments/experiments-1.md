---
title: "FER2013 Challenge"
excerpt: "Comparing different feature selection and classification techniques on facial expression data."
collection: portfolio
---

## CNN + SVM

* Here we try to examine the performance of a CNN classifier against using CNN as a feature extractor and using SVM as the final classifier.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

```python
class Data:
    def __init__(self, data):
        self._x = list(data[:, 1])
        self._y = data[:, 0]
        x_len = len(self._x[0])
        for xdx, x in enumerate(self._x):
            pixels = []
            lable = None
            for idx, i in enumerate(x.split(' ')):
                pixels.append(int(i))
            pixels = np.array(pixels).reshape((1, 48, 48))
            self._x[xdx] = pixels
            self._y[xdx] = int(self._y[xdx])
        self._x = np.array(self._x).reshape((len(self._x), 1, 48, 48))
        self._y = np.array(self._y)
```

```python
class FileReader:
    def __init__(self, csv_file_name):
        self._csv_file_name = csv_file_name
    def read(self):
        data = pd.read_csv(self._csv_file_name)
        self._data = data.values
```

```python
file_reader = FileReader('fer2013/fer2013.csv')
file_reader.read()
```

```python
data = Data(file_reader._data)
```

#### Preprocess the data

```python
data._x = np.asarray(data._x, dtype=np.float64)
data._x -= np.mean(data._x, axis = 0)
data._x /= np.std(data._x, axis = 0)
```

```python
for ix in range(10):
    plt.figure(ix)
    plt.imshow(data._x[ix].reshape((48, 48)), interpolation='none', cmap='gray')
plt.show()
```

```python
split_ratio = 0.8

train_indices = np.random.choice(len(data._x), int(len(data._x)*split_ratio))
test_indices = [i for i in range(len(data._x)) if i not in train_indices]

x_train, y_train = data._x[train_indices], data._y[train_indices]
x_valid, y_valid = data._x[test_indices], data._y[test_indices]
```

#### Implementation of CNN Architecture

```python
class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5),
            nn.PReLU(),
            nn.ZeroPad2d(2),
            nn.MaxPool2d(kernel_size=5, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.PReLU(),
            nn.ZeroPad2d(padding=1)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.PReLU()
        )

        self.layer5 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.PReLU(),
            nn.ZeroPad2d(1),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )

        self.fc1 = nn.Linear(3200, 1024)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 7)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.prelu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.prelu(x)
        x = self.dropout(x)

        y = self.fc3(x)
        return y
```

#### Dataset for pytorch DataLoader


```python
class FER2013Dataset(Dataset):
    """FER2013 Dataset."""

    def __init__(self, X, Y, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self._X = X
        self._Y = Y

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        return {'inputs': self._X[idx], 'labels': self._Y[idx]}
```

#### Network hyperparameters

```python
NUM_EPOCHS = 2
BATCH_SIZE = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

#### Create the train and test loader

```python
train_set = FER2013Dataset(x_train, y_train)
test_set = FER2013Dataset(x_valid, y_valid)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
```

#### Initialize the network and loss

```python
cnn = CNN()
cnn = cnn.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
```

#### Train the network

```python
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data['inputs'], data['labels']
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = cnn(inputs)
        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i%1000 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
print('Trainig complete')
```
