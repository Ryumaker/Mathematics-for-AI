from utils import load_checkpoint, make_grid, reverse_normalize
from model import CycleGenerator
import torch
from dataset import ImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

G_XtoY = CycleGenerator(conv_dim=64, n_res_blocks=6)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    G_XtoY.to(device)
    print('Models moved to GPU.')
else:
    print('Only CPU available.')

G_XtoY.load_state_dict(load_checkpoint("checkpoint/G_XtoY.pkl"))
G_XtoY.eval()

batch_size_test = 8
test_data = ImageDataset("dataset/test_img",img_size=256, normalize=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size_test, shuffle=False, num_workers=0, pin_memory=True)

test_iter = iter(test_dataloader)

X = test_iter.next()

fake_Y = G_XtoY(X.to(device))

X = make_grid(X, nrow=4).permute(1, 2, 0).detach().cpu().numpy()
X = reverse_normalize(X, 0.5, 0.5)
fake_Y = make_grid(fake_Y, nrow=4).permute(1, 2, 0).detach().cpu().numpy()
fake_Y = reverse_normalize(fake_Y, 0.5, 0.5)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(20, 10))

ax1.imshow(X)
ax1.axis('off')
ax1.set_title('X')
ax2.imshow(fake_Y)
ax2.axis('off')
ax2.set_title('Fake Y')
plt.show()
