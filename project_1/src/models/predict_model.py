from lightning_net import Net
from torchvision import transforms
# db imports:
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from project_1.src.visualization.visual_gates import gates_visuals


def get_latest_path(directory):
    """Returns path of the latest file(model in our case) writen in directory"""
    # Get list of all the files name in directory:
    files = os.listdir(directory)

    # From files names list creates os paths objects:
    paths = [os.sep.join([directory, basename]) for basename in files]

    # Sort files by time:
    paths.sort(key=os.path.getmtime, reverse=True)

    latest_file_path = paths[0]

    print("Latest file path:", latest_file_path)
    return latest_file_path


def pred():
    # Reading the latest model to evaluate:
    CKPT_PATH = get_latest_path("C:\\Users\\roizi\\PycharmProjects\\pythonProject2\\project_1\\models")
    latest_model = Net.load_from_checkpoint(CKPT_PATH)
    latest_model.eval()
    fig = gates_visuals(left_gate=latest_model.net.S_left_gate.get_gates().numpy()
                  , right_gate=latest_model.net.S_right_gate.get_gates().numpy())
    fig.show()

    # Dataset define:
    transform_X = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    mnist_path = "C:\\Users\\roizi\\PycharmProjects\\pythonProject2\\project_1\\data\\raw"
    X_ds = MNIST(mnist_path, download=True, transform=transform_X)

    # Dataloader:
    X_loader = DataLoader(X_ds, batch_size=128, num_workers=6)  # , shuffle=True

    predictions_x = torch.empty((len(X_ds), 10))
    true_labels_x = torch.empty((len(X_ds), 1))

    # Predictions evaluation using Kmeans++ algorithm:
    index = 0
    with torch.no_grad():
        for batch in X_loader:
            inputs, labels = batch

            outputs = latest_model.left_net_forward(inputs)

            predictions_x[index:(index + inputs.size(0)), :] = outputs.squeeze()
            true_labels_x[index:(index + labels.size(0))] = labels[:, None]
            index += inputs.size(0)

        kmeans = KMeans(n_clusters=10, n_init='auto').fit(predictions_x)

        # calculate the success rate of the KMeans model using the accuracy_score function
        success_rate = accuracy_score(y_true=true_labels_x, y_pred=kmeans.labels_)
        print("Success rate: {:.2f}%".format(success_rate * 100))


if __name__ == "__main__":
    pred()


