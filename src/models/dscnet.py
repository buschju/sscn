from typing import List, Tuple

import mlflow
import numpy
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.init import constant_
from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset

from eval_utils import post_process_coefficient_matrix
from models.utils import get_loss_reconst, get_loss_reg, get_loss_ssc, get_conv_autoencoder, \
    get_identity_autoencoder, ClusteringAutoencoder


class DscNet(ClusteringAutoencoder):
    def __init__(self,
                 encoder: Module,
                 decoder: Module,
                 batch_size: int,
                 lambda_reconst: float,
                 lambda_reg: float,
                 lambda_ssc: float,
                 ):
        super(DscNet, self).__init__(encoder=encoder,
                                     decoder=decoder,
                                     )

        self.lambda_reconst = lambda_reconst
        self.lambda_reg = lambda_reg
        self.lambda_ssc = lambda_ssc

        self.se_layer = SeLayer(batch_size=batch_size)

        self.reset_parameters()

    def train_step_dscnet(self,
                          X: Tensor,
                          optimizer: Optimizer,
                          ) -> Tuple[float, float, float, float]:
        self.train()

        # Forward pass
        Z = self.encoder(X)
        Z_se = self.se_layer(Z)
        X_out = self.decoder(Z_se)

        # Get losses
        loss_reconst = get_loss_reconst(X, X_out)
        loss_reg = get_loss_reg(self.se_layer.get_coefficients())
        loss_ssc = get_loss_ssc(Z, Z_se)
        loss_se = 0.
        if self.lambda_reconst > 0:
            loss_se += self.lambda_reconst * loss_reconst
        if self.lambda_reg > 0:
            loss_se += self.lambda_reg * loss_reg
        if self.lambda_ssc > 0:
            loss_se += self.lambda_ssc * loss_ssc

        # Backward pass
        optimizer.zero_grad()
        loss_se.backward()
        optimizer.step()

        return loss_reconst.item(), loss_reg.item(), loss_ssc.item(), loss_se.item()

    def train_dscnet(self,
                     dataset: Dataset,
                     num_epochs: int,
                     learning_rate: float,
                     ) -> None:
        optimizer = Adam(params=self.parameters(),
                         lr=learning_rate,
                         )
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=len(dataset),
                                                 drop_last=False,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 pin_memory=True,
                                                 )

        for epoch in range(1, num_epochs + 1):
            for batch in dataloader:
                X = batch[0].to('cuda')
                losses_epoch = self.train_step_dscnet(X,
                                                      optimizer,
                                                      )
                mlflow.log_metric('loss_reconst', losses_epoch[0], step=epoch)
                mlflow.log_metric('loss_reg', losses_epoch[1], step=epoch)
                mlflow.log_metric('loss_ssc', losses_epoch[2], step=epoch)
                mlflow.log_metric('loss_se', losses_epoch[3], step=epoch)

    def train_model(self,
                    dataset: Dataset,
                    num_epochs: int,
                    learning_rate: float,
                    ) -> None:
        # Optimize all parameters jointly
        self.train_dscnet(dataset=dataset,
                          num_epochs=num_epochs,
                          learning_rate=learning_rate,
                          )

    def get_cluster_assignments(self,
                                num_clusters: int,
                                max_cluster_dim: int,
                                noise_threshold: float,
                                noise_alpha: float,
                                dataset: Dataset = None,
                                ) -> numpy.ndarray:
        self.eval()

        y_pred, _ = post_process_coefficient_matrix(self.se_layer.get_coefficients().cpu().numpy(),
                                                    num_clusters=num_clusters,
                                                    max_cluster_dim=max_cluster_dim,
                                                    noise_threshold=noise_threshold,
                                                    noise_alpha=noise_alpha,
                                                    )

        return y_pred

    def reset_parameters(self):
        self.reset_autoencoder()
        self.se_layer.reset_parameters()


class DscNetLinear(DscNet):
    def __init__(self,
                 batch_size: int,
                 lambda_reg: float,
                 lambda_ssc: float,
                 ):
        encoder, decoder = get_identity_autoencoder()

        super(DscNetLinear, self).__init__(encoder=encoder,
                                           decoder=decoder,
                                           batch_size=batch_size,
                                           lambda_reconst=0,
                                           lambda_reg=lambda_reg,
                                           lambda_ssc=lambda_ssc,
                                           )


class DscNetConv(DscNet):
    def __init__(self,
                 image_size: Tuple[int, int],
                 in_channels: int,
                 kernel_size: List[int],
                 num_channels: List[int],
                 stride: int,
                 padding: List[int],
                 batch_size: int,
                 lambda_reconst: float,
                 lambda_reg: float,
                 lambda_ssc: float,
                 ):
        # Autoencoder
        encoder, decoder, _ = get_conv_autoencoder(image_size=image_size,
                                                   in_channels=in_channels,
                                                   kernel_size=kernel_size,
                                                   num_channels=num_channels,
                                                   stride=stride,
                                                   padding=padding,
                                                   )

        super(DscNetConv, self).__init__(encoder=encoder,
                                         decoder=decoder,
                                         batch_size=batch_size,
                                         lambda_reconst=lambda_reconst,
                                         lambda_reg=lambda_reg,
                                         lambda_ssc=lambda_ssc,
                                         )


class SeLayer(Module):
    def __init__(self,
                 batch_size: int,
                 ):
        super().__init__()
        self.C = Parameter(1e-8 * torch.ones(batch_size, batch_size),
                           requires_grad=True,
                           )
        self.reset_parameters()

    def forward(self,
                X: Tensor,
                ) -> Tensor:
        return self.C @ X

    def get_coefficients(self,
                         ) -> Tensor:
        return self.C.data

    def reset_parameters(self):
        constant_(self.C, 1e-8)
