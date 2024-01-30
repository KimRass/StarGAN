import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_encode_label(label, n_classes):
    return torch.eye(n_classes)[label]


# "where [ ] refers to concatenation, and ci represents a vector for the labels of the i-th dataset. The vector of the known label ci can be represented as either a binary vector for binary attributes or a one-hot vector for categorical attributes. For the remaining n􀀀1 unknown labels we simply assign zero values. In our experiments, we utilize the CelebA and RaFD datasets, where n is two."

batch_size = 4
n_doms1 = 7
n_classes2 = 8
n_datasets = 2

label1 = torch.randint(low=0, high=2, size=(batch_size, n_doms1), dtype=torch.float32)
label2 = torch.randint(low=0, high=n_classes2 - 1, size=(batch_size,))
label2 = one_hot_encode_label(label=label2, n_classes=n_classes2)
mask = torch.randint(low=0, high=2, size=(batch_size,))
mask = one_hot_encode_label(label=mask, n_classes=n_datasets)
dom_label = torch.cat([label1, label2, mask], dim=1)
dom_label.shape
