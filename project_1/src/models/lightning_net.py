import pytorch_lightning as pl
import torch
from sdcca import MyModel
import os


class Net(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.save_hyperparameters()
        self.hp = hp

        # pytorch net object:
        self.net = MyModel(hp)

        self.C_t_1 = torch.zeros(self.hp.layers.out_layer)  # , device="cuda:0")

        self.N_factor = 0  # shown as ct in the article

        #  gates:
        path = os.path.dirname(os.path.realpath(__file__))
        self.path_to_images = os.path.join(os.path.dirname(path), 'visualization')

    def metric(self, X_hat, Y_hat):
        gates_loss = self.net.S_left_gate.get_reg() + self.net.S_right_gate.get_reg()
        return - self.correlation_sdl(X_hat.squeeze(), Y_hat.squeeze()) + gates_loss

    def correlation_sdl(self, X, Y):
        # normalization factor according to "Scalable and Effective Deep CCA via Soft Decorrelation" article:
        self.N_factor = self.hp.alpha * self.N_factor + 1

        m = X.size(dim=1)
        C_mini = torch.matmul(torch.t(Y), X) / m

        self.C_t_1 = (self.hp.alpha * self.C_t_1.detach() + C_mini) / self.N_factor
        corr_sdl = torch.sum(torch.abs((self.C_t_1 + C_mini).fill_diagonal_(0)))
        return corr_sdl

    def forward(self, batch):
        [X, x_label], [Y, y_label] = batch
        left_gate = torch.load(os.path.join(self.path_to_images, 'left_gate.pt'))
        right_gate = torch.load(os.path.join(self.path_to_images, 'right_gate.pt'))
        X_out = self.net.left_net(X * left_gate)
        Y_out = self.net.right_net(Y * right_gate)
        return X_out, x_label, Y_out, y_label

    def training_step(self, batch):
        [X, _], [Y, _] = batch
        X_hat, Y_hat = self.net(X, Y)

        loss = self.metric(X_hat, Y_hat)

        # tensorboard logs:
        tensorboard_logs = {'train_loss': loss}
        self.log("loss", loss, prog_bar=True, logger=True)
        return {'loss': loss, 'log': tensorboard_logs}

    def on_train_epoch_start(self) -> None:
        self.C_t_1 = torch.zeros(self.hp.layers.out_layer)

    def on_train_epoch_end(self) -> None:
        torch.save(self.net.S_left_gate.mus.detach(), os.path.join(self.path_to_images, 'left_gate.pt'))
        torch.save(self.net.S_right_gate.mus.detach(), os.path.join(self.path_to_images, 'right_gate.pt'))

    """
    def test_step(self, batch):
        [X, x_label], [Y, y_label] = batch
        left_gate = torch.load(os.path.join(self.path_to_images, 'left_gate.pt'))
        right_gate = torch.load(os.path.join(self.path_to_images, 'right_gate.pt'))
        X_out = self.net.left_net(X * left_gate)
        Y_out = self.net.right_net(Y * right_gate)
        return X_out, x_label, Y_out, y_label
    """

    def configure_optimizers(self):
        # default optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hp.lr)
        if self.hp == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hp.lr)
        return {'optimizer': optimizer}
