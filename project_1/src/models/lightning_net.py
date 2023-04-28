import pytorch_lightning as pl
import torch
from sdcca import MyModel
import os
from torchvision.utils import make_grid


class Net(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        #  self.automatic_optimization = False
        self.save_hyperparameters()
        self.hp = hp

        # pytorch net object:
        self.net = MyModel(hp)

        self.C_t_1 = torch.zeros(self.hp.layers.out_layer)  # can be problematic with gpu run

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

        m = X.size(dim=0)
        C_mini = torch.matmul(torch.t(Y), X) / m

        self.C_t_1 = (self.hp.alpha * self.C_t_1.detach() + C_mini)
        corr_sdl = torch.sum(torch.abs(self.C_t_1 / self.N_factor))
        return corr_sdl  # + torch.sum(torch.abs(C_mini))

    def right_net_forward(self, Y):
        if self.hp.gates_learning:
            right_gate = self.net.S_right_gate.get_gates()
            Y_out = self.net.right_net(Y * right_gate)
        else:
            Y_out = self.self.net.left_net(Y)
        return Y_out

    def left_net_forward(self, X):
        if self.hp.gates_learning:
            left_gate = self.net.S_left_gate.get_gates()
            X_out = self.net.left_net(X * left_gate)
        else:
            X_out = self.self.net.left_net(X)
        return X_out

    def training_step(self, batch):
        [X, _], [Y, _] = batch
        X_hat, Y_hat = self.net(X, Y)
        loss = self.metric(X_hat, Y_hat)

        #  gate_opt1, gate_opt2, opt3, opt4 = self.optimizers()

        # tensorboard logs:
        self.log("loss", loss, prog_bar=True, logger=True)
        return {'loss': loss}

    def on_train_epoch_start(self) -> None:
        self.N_factor = 0
        self.C_t_1.new_zeros(self.hp.layers.out_layer)

    def on_train_epoch_end(self) -> None:
        # In case that we activate gates parameters learning visual gates with tensorboard:
        if self.hp.gates_learning:
            # Convert the example image to a grid
            left_gate_grid = make_grid(self.net.S_left_gate.mus)
            right_gate_grid = make_grid(self.net.S_right_gate.mus)
            # Log the image to TensorBoard
            self.logger.experiment.add_image("Left Gate mus pram", left_gate_grid, self.current_epoch)
            self.logger.experiment.add_image("Right Gate mus pram", right_gate_grid, self.current_epoch)
            self.logger.experiment.add_image("Left Gate", make_grid(self.net.S_left_gate.get_gates()),
                                             self.current_epoch)
            self.logger.experiment.add_image("Right Gate", make_grid(self.net.S_right_gate.get_gates()),
                                             self.current_epoch)

    def configure_optimizers(self):
        # default optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hp.lr)

        # left_optimizer = torch.optim.Adam(self.net.S_left_gate.parameters(), lr=self.hp.lr)
        # right_optimizer = torch.optim.Adam(self.net.S_right_gate.parameters(), lr=self.hp.lr)
        # dim_reduction_left_optimizer = torch.optim.Adam(self.net.left_net.parameters(), lr=self.hp.lr)
        # dim_reduction_right_optimizer = torch.optim.Adam(self.net.right_net.parameters(), lr=self.hp.lr)

        if self.hp == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hp.lr)

        return {'optimizer': optimizer}
