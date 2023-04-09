from lightning_net import Net
import matplotlib.pyplot as plt
from torchvision import transforms
# db imports:
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def pred():
    CKPT_PATH = "C:/Users/roizi/PycharmProjects/pythonProject2/project_1/models/checkpoint_epoch=02-loss=-400.33.ckpt"
    bests_model = Net.load_from_checkpoint(CKPT_PATH).eval()
    transform_X = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    transform_Y = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    X_ds = MNIST("C:/Users/roizi/PycharmProjects/pythonProject2/project_1/data/raw", download=True, transform=transform_X)
    Y_ds = MNIST("C:/Users/roizi/PycharmProjects/pythonProject2/project_1/data/raw", download=True, transform=transform_Y)

    # dataloaders
    X_loader = DataLoader(X_ds, batch_size=128, num_workers=6)  # , shuffle=True
    Y_loader = DataLoader(Y_ds, batch_size=128, num_workers=6)  # , shuffle=True

    X_out, x_label, Y_out, y_label = bests_model(X_loader, Y_loader)
    print(X_out)

    fig, ax = plt.subplots()

    # left gate fig:
    ax.imshow(X_out.detach().numpy())

    plt.show()


if __name__ == "__main__":
    pred()
