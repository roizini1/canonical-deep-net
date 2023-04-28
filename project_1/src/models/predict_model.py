from lightning_net import Net
from torchvision import transforms
# db imports:
import os
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from project_1.src.visualization.visual_gates import gates_visuals


def most_common_number(tensor):
    """
    Returns the number that repeats the most in a PyTorch tensor along with its count.

    Args:
        tensor (torch.Tensor): Input PyTorch tensor.

    Returns:
        tuple: Tuple containing the most common number (int) and its count (int).
    """
    # Flatten the tensor
    flattened_tensor = tensor.view(-1)

    # Count the occurrences of each number
    unique_numbers, counts = torch.unique(flattened_tensor, return_counts=True)

    # Get the index of the maximum count
    max_count_idx = torch.argmax(counts)

    # Get the most common number and its count
    most_common_number = unique_numbers[max_count_idx].item()
    most_common_count = counts[max_count_idx].item()

    return most_common_number, most_common_count


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
    X_ds = MNIST(mnist_path, train=False, download=True, transform=transform_X)

    # Dataloader:
    X_loader = DataLoader(X_ds, batch_size=128, num_workers=6)  # , shuffle=True

    predictions_x = torch.empty((len(X_ds), 10))
    true_labels_x = torch.empty((len(X_ds), 1))

    # Predictions evaluation using Kmeans++ algorithm:
    index = 0
    with torch.no_grad():
        for batch in X_loader:
            inputs, labels = batch

            outputs = latest_model.right_net_forward(inputs)

            predictions_x[index:(index + inputs.size(0)), :] = outputs.squeeze()
            true_labels_x[index:(index + labels.size(0))] = labels[:, None]
            index += inputs.size(0)

        kmeans = KMeans(n_clusters=10, init='k-means++', n_init='auto').fit(X=predictions_x, y=true_labels_x)
        most_common, most_common_count = most_common_number(true_labels_x[kmeans.labels_ == 0])
        print(most_common, most_common_count)
        print(torch.sum(torch.where(torch.tensor(kmeans.labels_) == most_common, 1, 0)))

        # calculate the success rate of the KMeans model using the accuracy_score function
        success_rate = accuracy_score(y_true=true_labels_x, y_pred=kmeans.labels_)
        print("Success rate: {:.2f}%".format(success_rate * 100))


if __name__ == "__main__":
    pred()
