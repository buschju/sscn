from typing import Tuple, List, Union

import mlflow
import numpy
import torch
from torch import Tensor
from torch.nn import Module, Conv2d, ReLU, Sequential, ConvTranspose2d, Linear
from torch.nn.functional import cross_entropy
from torch.nn.init import zeros_, xavier_uniform_
from torch.nn.modules.linear import Identity
from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset


def get_loss_reconst(X: Tensor,
                     X_out: Tensor,
                     ) -> Tensor:
    return torch.sum((X - X_out) ** 2)


def get_loss_reg(Q: Tensor,
                 ) -> Tensor:
    return torch.sum(Q ** 2)


def get_loss_ssc(Z: Tensor,
                 Z_se: Tensor,
                 ) -> Tensor:
    return torch.sum((Z - Z_se) ** 2) / 2.


def get_loss_assignment(A_s: Tensor,
                        A_c: Tensor,
                        ) -> Tensor:
    return ((A_s - A_c) ** 2).sum()


def get_loss_triplet_hard(A_c: Tensor,
                          pos: Tensor,
                          non_pos: Tensor,
                          neg: Tensor,
                          margin: int,
                          ) -> Tensor:
    hard_positives = (pos * A_c + non_pos).min(dim=1)[0]
    hard_negatives = (neg * A_c).max(dim=1)[0]

    loss = hard_negatives - hard_positives + margin
    loss = torch.clamp(loss,
                       min=0.,
                       )
    return loss.sum()


def get_loss_supervised(outputs: Tensor,
                        y: Tensor,
                        ):
    return cross_entropy(outputs, y)


def get_projection_distance(X: Tensor,
                            S: Tensor,
                            ) -> Tensor:
    HS = torch.einsum('nd,kdp->knp', X, S)
    HSSt = torch.einsum('knp,kdp->knd', HS, S)

    return ((HSSt - X) ** 2).sum(dim=-1).T


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                x: Tensor,
                ) -> Tensor:
        return x.view(x.size(0), -1)


class Unflatten(Module):
    def __init__(self,
                 image_size: Tuple[int, int],
                 num_channels: int,
                 ):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels

    def forward(self,
                x: Tensor,
                ) -> Tensor:
        return x.view(x.size(0),
                      self.num_channels,
                      self.image_size[0],
                      self.image_size[1],
                      )


def conv_output_shape(image_size: Union[int, Tuple[int, int]],
                      kernel_size: Union[int, Tuple[int, int]] = 1,
                      stride: Union[int, Tuple[int, int]] = 1,
                      padding: Union[int, Tuple[int, int]] = 0,
                      dilation: int = 1,
                      ) -> Tuple[int, int]:
    """
    Source: https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/6
    """
    if type(image_size) is int:
        image_size = (image_size, image_size)

    if type(kernel_size) is int:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is int:
        stride = (stride, stride)

    if type(padding) is int:
        padding = (padding, padding)

    image_size_out = tuple(
        (image_size[i] + (2 * padding[i]) - (dilation * (kernel_size[i] - 1)) - 1) // stride[i] + 1
        for i in range(2)
    )

    return image_size_out


def get_conv_autoencoder(image_size: Tuple[int, int],
                         in_channels: int,
                         kernel_size: List[int],
                         num_channels: List[int],
                         stride: int,
                         padding: List[int],
                         ) -> Tuple[Module, Module, int]:
    # Build encoder
    layers_encoder = [
        Conv2d(in_channels=in_channels,
               out_channels=num_channels[0],
               kernel_size=(kernel_size[0], kernel_size[0]),
               stride=stride,
               padding=padding[0],
               bias=True,
               ),
        ReLU(),
    ]
    for i in range(1, len(kernel_size)):
        layers_encoder += [
            Conv2d(in_channels=num_channels[i - 1],
                   out_channels=num_channels[i],
                   kernel_size=(kernel_size[i], kernel_size[i]),
                   stride=stride,
                   padding=padding[i],
                   bias=True,
                   ),
            ReLU(),
        ]
    layers_encoder += [Flatten()]
    encoder = Sequential(*layers_encoder)

    # Get intermediate image sizes in encoder
    image_size_enc = [image_size]
    for i in range(len(kernel_size)):
        image_size_enc += [
            conv_output_shape(image_size_enc[i],
                              kernel_size=kernel_size[i],
                              padding=padding[i],
                              stride=stride,
                              ),
        ]
    hidden_size = image_size_enc[-1][0] * image_size_enc[-1][1] * num_channels[-1]

    # Build decoder
    output_padding = [
        tuple(
            -((image_size_enc[i][j] - 1) * stride - 2 * padding[i - 1] + kernel_size[i - 1] - image_size_enc[i - 1][j])
            for j in range(2)
        )
        for i in range(1, len(image_size_enc))
    ]
    layers_decoder = [
        ReLU(),
        ConvTranspose2d(in_channels=num_channels[0],
                        out_channels=in_channels,
                        kernel_size=(kernel_size[0], kernel_size[0]),
                        stride=stride,
                        padding=padding[0],
                        output_padding=output_padding[0],
                        bias=True,
                        ),
    ]

    for i in range(1, len(num_channels)):
        layers_decoder += [
            ReLU(),
            ConvTranspose2d(in_channels=num_channels[i],
                            out_channels=num_channels[i - 1],
                            kernel_size=(kernel_size[i], kernel_size[i]),
                            stride=stride,
                            padding=padding[i],
                            output_padding=output_padding[i],
                            bias=True,
                            ),
        ]
    layers_decoder += [Unflatten(image_size_enc[-1], num_channels[-1])]
    decoder = Sequential(*layers_decoder[::-1])

    return encoder, decoder, hidden_size


def get_identity_autoencoder() -> Tuple[Module, Module]:
    encoder = Sequential(Identity())
    decoder = Sequential(Identity())

    return encoder, decoder


class ClusteringAutoencoder(Module):
    def __init__(self,
                 encoder: Module,
                 decoder: Module,
                 ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def pretrain_step_autoencoder(self,
                                  X: Tensor,
                                  optimizer: Optimizer,
                                  ) -> float:
        self.train()

        # Forward pass
        Z = self.encoder(X)
        X_out = self.decoder(Z)

        # Get losses
        loss_reconst = get_loss_reconst(X, X_out)

        # Backward pass
        optimizer.zero_grad()
        loss_reconst.backward()
        optimizer.step()

        return loss_reconst.item()

    def pretrain_autoencoder(self,
                             dataset: Dataset,
                             num_epochs: int,
                             learning_rate: float,
                             batch_size: int,
                             shuffle_batches: bool,
                             ) -> None:
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = Adam(params=parameters,
                         lr=learning_rate,
                         )

        if batch_size is None:
            batch_size = len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 drop_last=False,
                                                 shuffle=shuffle_batches,
                                                 num_workers=0,
                                                 pin_memory=True,
                                                 )

        for epoch in range(1, num_epochs + 1):
            loss_epoch = 0
            num_batches = 0

            for batch in dataloader:
                X_batch = batch[0].to('cuda')
                loss = self.pretrain_step_autoencoder(X_batch, optimizer)
                loss_epoch += loss
                num_batches += 1

            loss_epoch /= num_batches
            mlflow.log_metric('loss_reconst_autoencoder', loss_epoch, step=epoch)

    def reset_autoencoder(self) -> None:
        for module in [self.encoder, self.decoder]:
            for layer in module:
                reset_layer(layer)

    def get_cluster_assignments(self,
                                dataset: Dataset = None,
                                **kwargs,
                                ) -> numpy.ndarray:
        raise NotImplementedError()

    def reset_parameters(self) -> None:
        self.reset_autoencoder()

    def train_model(self,
                    dataset: Dataset,
                    pretrain_autoencoder: bool,
                    **kwargs,
                    ) -> None:
        raise NotImplementedError()

    def __repr__(self):
        return self.__class__.__name__


def reset_layer(layer: Module,
                ) -> None:
    if isinstance(layer, Conv2d) \
            or isinstance(layer, ConvTranspose2d) \
            or isinstance(layer, Linear):
        xavier_uniform_(layer.weight)
        if layer.bias is not None:
            zeros_(layer.bias)

    elif isinstance(layer, Identity):
        pass
    elif isinstance(layer, ReLU):
        pass
    elif isinstance(layer, Flatten):
        pass
    elif isinstance(layer, Unflatten):
        pass
    else:
        raise ValueError('Unknown layer: {}'.format(layer.__class__.__name__))
