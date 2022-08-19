import torch
x = torch.ones(1, 512, 11025)
cat_term = torch.ones(1, 512, 11026)
print(x.shape)
diff = cat_term.shape[-1] - x.shape[-1]
x = torch.nn.functional.pad(x, (diff, 0), mode='constant', value=0)
print(x.shape)
print(x)

g = torch.randn(1, 5)
print(g)
g = torch.nn.functional.pad(g, (0, -3), mode='constant', value=0)
print(g)