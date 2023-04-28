import matplotlib.pyplot as plt


def gates_visuals(left_gate, right_gate):
    """This function is loading the saved mus parameter for each gate and returns the figure to display/save"""
    # figures declaration:
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Gates', fontsize=24)

    # left gate fig:
    im1 = ax[0].imshow(left_gate)
    ax[0].set(title="Left gate")
    ax[0].axis('off')
    fig.colorbar(im1, fraction=0.047)

    # right gate fig:
    im2 = ax[1].imshow(right_gate)
    ax[1].set(title="Right gate")
    ax[1].axis('off')
    fig.colorbar(im2, fraction=0.047)

    return fig
