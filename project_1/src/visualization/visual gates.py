import matplotlib.pyplot as plt
import torch


def gates_visuals():
    # base file dir
    left_gate = torch.load('left_gate.pt')
    right_gate = torch.load('right_gate.pt')

    # figures declaration:
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Gates', fontsize=24)

    # left gate fig:
    im1 = ax[0].imshow(left_gate.numpy())
    ax[0].set(title="Left gate")
    ax[0].axis('off')
    plt.colorbar(im1, fraction=0.047)

    # right gate fig:
    im2 = ax[1].imshow(right_gate.numpy())
    ax[1].set(title="Right gate")
    ax[1].axis('off')
    plt.colorbar(im2, fraction=0.047)

    # save to image:
    plt.savefig("Gates visualization")

    plt.show()


if __name__ == "__main__":
    gates_visuals()
