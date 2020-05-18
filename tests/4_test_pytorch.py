import torch

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

x = torch.rand(5, 3)
print(x)

