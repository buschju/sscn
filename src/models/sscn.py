import abc
from copy import deepcopy
from typing import List, Tuple, Optional

import mlflow
import numpy
import torch
from torch import Tensor
from torch.nn import Module, Linear, Parameter
from torch.nn.init import eye_, zeros_, xavier_uniform_
from torch.optim import Optimizer, Adam
from torch.utils.data import TensorDataset, Dataset

from eval_utils import post_process_coefficient_matrix, evaluate_model
from models.utils import get_loss_reconst, get_loss_reg, get_loss_ssc, get_loss_triplet_hard, get_conv_autoencoder, \
    ClusteringAutoencoder, get_identity_autoencoder, get_projection_distance, get_loss_supervised
from stiefel_optimizer.stiefel_optimizer import AdamG


class Sscn(ClusteringAutoencoder):
    def __init__(self,
                 encoder: Module,
                 decoder: Module,
                 classifier: Module,
                 hidden_size: int,
                 num_clusters: int,
                 max_cluster_dim: int,
                 lambda_reconst: float,
                 lambda_reg: float,
                 lambda_ssc: float,
                 lambda_assignment: float,
                 se_bias: bool,
                 ):
        super(Sscn, self).__init__(encoder=encoder,
                                   decoder=decoder,
                                   )

        self.num_clusters = num_clusters
        self.max_cluster_dim = max_cluster_dim
        self.lambda_reconst = lambda_reconst
        self.lambda_reg = lambda_reg
        self.lambda_ssc = lambda_ssc
        self.lambda_assignment = lambda_assignment

        hidden_size_se = num_clusters * max_cluster_dim
        self.se_encoder = Linear(in_features=hidden_size,
                                 out_features=hidden_size_se,
                                 bias=se_bias,
                                 )

        self.classifier = classifier

        self.reset_parameters()

    @staticmethod
    def get_self_expression_affinities(Q: Tensor,
                                       ) -> Tensor:
        Q_abs = torch.abs(Q)
        Q_max = Q_abs.detach().clone().fill_diagonal_(0).max(dim=0, keepdim=True)[
            0]  # column-wise max since our data matrices are n x d
        Q_max[Q_max == 0.] = 1.
        A_s = Q_abs / Q_max
        A_s = (A_s + A_s.T) / 2.
        A_s = A_s.fill_diagonal_(1.)

        return A_s

    def get_classification_affinities(self,
                                      H: Tensor,
                                      ) -> Tensor:
        # soft_assignments = normalize(self.classifier.get_soft_assignments(H),
        #                              dim=1,
        #                              p=2.,
        #                              )
        soft_assignments = self.classifier.get_soft_assignments(H)

        A_c = soft_assignments @ soft_assignments.T

        return A_c

    def get_cluster_assignments(self,
                                dataset: Dataset,
                                method: str,
                                **kwargs,
                                ) -> numpy.ndarray:
        self.eval()

        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        else:
            batch_size = None
        if batch_size is None:
            batch_size = len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 drop_last=False,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 pin_memory=True,
                                                 )

        with torch.no_grad():
            if method == 'classifier':
                soft_assignments = numpy.concatenate([self.classifier.get_soft_assignments(
                    self.se_encoder(self.encoder(batch[0].to('cuda')))).detach().cpu().numpy() for batch in dataloader],
                                                     axis=0)
                y_pred = numpy.argmax(soft_assignments, axis=1)
            elif method == 'spectral_clustering':
                H = numpy.concatenate(
                    [self.se_encoder(self.encoder(batch[0].to('cuda'))).detach().cpu().numpy() for batch in dataloader],
                    axis=0)
                Q = H @ H.T
                y_pred, _ = post_process_coefficient_matrix(Q,
                                                            num_clusters=self.num_clusters,
                                                            max_cluster_dim=self.max_cluster_dim,
                                                            noise_threshold=kwargs['noise_threshold'],
                                                            noise_alpha=kwargs['noise_alpha'],
                                                            )
            else:
                raise ValueError('Unknown cluster assignment method: {}'.format(method))

        return y_pred

    def train_step_se(self,
                      X: Tensor,
                      optimizer: Optimizer,
                      ) -> Tuple[float, float, float, float]:
        self.train()

        # Forward pass
        Z = self.encoder(X)
        H = self.se_encoder(Z)
        Q = H @ H.T
        Z_se = Q @ Z
        X_out = self.decoder(Z_se)

        # Get losses
        loss_reconst = get_loss_reconst(X, X_out)
        loss_reg = get_loss_reg(Q)
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

    def train_se(self,
                 dataset: Dataset,
                 num_epochs: int,
                 learning_rate: float,
                 batch_size: int,
                 shuffle_batches: bool,
                 ) -> None:
        parameters = list(self.encoder.parameters()) \
                     + list(self.decoder.parameters()) \
                     + list(self.se_encoder.parameters())
        optimizer = Adam(params=parameters,
                         lr=learning_rate,
                         )

        if batch_size is None:
            batch_size = len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 drop_last=True,
                                                 shuffle=shuffle_batches,
                                                 num_workers=0,
                                                 pin_memory=True,
                                                 )

        for epoch in range(num_epochs):
            losses_epoch = numpy.zeros(4)
            num_batches = 0

            for batch in dataloader:
                X_batch = batch[0].to('cuda')
                losses = self.train_step_se(X_batch,
                                            optimizer,
                                            )
                losses = numpy.array(losses)
                losses_epoch += losses
                num_batches += 1

            losses_epoch /= num_batches
            mlflow.log_metric('loss_reconst', losses_epoch[0], step=epoch)
            mlflow.log_metric('loss_reg', losses_epoch[1], step=epoch)
            mlflow.log_metric('loss_ssc', losses_epoch[2], step=epoch)
            mlflow.log_metric('loss_se', losses_epoch[3], step=epoch)

    def train_step_classifier_pseudo_labels(self,
                                            X: Tensor,
                                            optimizer: Optimizer,
                                            num_iterations_per_batch: int,
                                            pseudo_labels: Tensor,
                                            ) -> float:
        self.train()

        # Forward pass
        Z = self.encoder(X)
        H = self.se_encoder(Z).detach()

        for _ in range(num_iterations_per_batch):
            outputs = self.classifier.get_outputs(H)
            loss_assignment = get_loss_supervised(outputs=outputs,
                                                  y=pseudo_labels,
                                                  )

            # Backward pass
            optimizer.zero_grad()
            loss_assignment.backward()
            optimizer.step()

        return loss_assignment.item()

    def train_step_classifier_triplets(self,
                                       X: Tensor,
                                       optimizer: Optimizer,
                                       num_iterations_per_batch: int,
                                       pos_thres: Optional[float],
                                       neg_thres: float,
                                       margin: float,
                                       ) -> float:
        self.train()

        # Forward pass
        Z = self.encoder(X)
        H = self.se_encoder(Z).detach()
        Q = H @ H.T
        A_s = self.get_self_expression_affinities(Q)

        pos_s = (A_s > pos_thres)
        non_pos_s = (A_s <= pos_thres)
        neg_s = (A_s < neg_thres)

        for _ in range(num_iterations_per_batch):
            A_c = self.get_classification_affinities(H)

            # Get loss
            loss_assignment = get_loss_triplet_hard(A_c=A_c,
                                                    pos=pos_s,
                                                    non_pos=non_pos_s,
                                                    neg=neg_s,
                                                    margin=margin,
                                                    )

            # Backward pass
            optimizer.zero_grad()
            loss_assignment.backward()
            optimizer.step()

        return loss_assignment.item()

    def train_classifier(self,
                         dataset: Dataset,
                         num_epochs: int,
                         num_iterations_per_batch: int,
                         learning_rate: float,
                         batch_size: int,
                         shuffle_batches: bool,
                         use_pseudo_labels: bool,
                         pos_thres: Optional[float] = None,
                         neg_thres: Optional[float] = None,
                         margin: Optional[float] = None,
                         noise_threshold: Optional[float] = None,
                         noise_alpha: Optional[float] = None,
                         ) -> None:
        optimizer = self.classifier.get_optimizer(learning_rate=learning_rate)

        if use_pseudo_labels:
            pseudo_labels = self.get_cluster_assignments(dataset=dataset,
                                                         method='spectral_clustering',
                                                         noise_threshold=noise_threshold,
                                                         noise_alpha=noise_alpha,
                                                         batch_size=batch_size,
                                                         )
            pseudo_labels = torch.LongTensor(pseudo_labels - 1)
            dataset_training = deepcopy(dataset)
            dataset_training.targets = pseudo_labels
        else:
            dataset_training = dataset

        if batch_size is None:
            batch_size = len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset_training,
                                                 batch_size=batch_size,
                                                 drop_last=True,
                                                 shuffle=shuffle_batches,
                                                 num_workers=0,
                                                 pin_memory=True,
                                                 )

        for epoch in range(1, num_epochs + 1):
            loss_epoch = 0
            num_batches = 0

            for batch in dataloader:
                X_batch = batch[0].to('cuda')
                if use_pseudo_labels:
                    pseudo_labels_batch = batch[1].to('cuda')
                    loss = self.train_step_classifier_pseudo_labels(X_batch,
                                                                    optimizer,
                                                                    num_iterations_per_batch=num_iterations_per_batch,
                                                                    pseudo_labels=pseudo_labels_batch,
                                                                    )
                else:
                    loss = self.train_step_classifier_triplets(X_batch,
                                                               optimizer,
                                                               num_iterations_per_batch=num_iterations_per_batch,
                                                               pos_thres=pos_thres,
                                                               neg_thres=neg_thres,
                                                               margin=margin,
                                                               )
                loss /= batch_size
                loss_epoch += loss
                num_batches += 1

            loss_epoch /= num_batches
            mlflow.log_metric('loss_assignment', loss_epoch, step=epoch)

            # Evaluate classification accuracy on whole dataset
            acc_epoch, ari_epoch, nmi_epoch = evaluate_model(model=self,
                                                             dataset=dataset,
                                                             method='classifier',
                                                             )
            mlflow.log_metric('acc_epoch', acc_epoch, step=epoch)
            mlflow.log_metric('ari_epoch', ari_epoch, step=epoch)
            mlflow.log_metric('nmi_epoch', nmi_epoch, step=epoch)

    def train_model(self,
                    dataset: Dataset,
                    num_epochs: int,
                    learning_rate: float,
                    batch_size: int,
                    shuffle_batches: bool,
                    ) -> None:
        # Train self-expressive model without classifier
        self.train_se(dataset=dataset,
                      num_epochs=num_epochs,
                      learning_rate=learning_rate,
                      batch_size=batch_size,
                      shuffle_batches=shuffle_batches,
                      )

    def reset_parameters(self):
        self.reset_autoencoder()
        self.reset_se_layer()
        self.classifier.reset_parameters()

    def reset_se_layer(self):
        xavier_uniform_(self.se_encoder.weight)
        if self.se_encoder.bias is not None:
            zeros_(self.se_encoder.bias)


class SscnLinear(Sscn):
    def __init__(self,
                 num_features: int,
                 num_clusters: int,
                 max_cluster_dim: int,
                 lambda_reg: float,
                 lambda_ssc: float,
                 lambda_assignment: float,
                 se_bias: bool,
                 ):
        encoder, decoder = get_identity_autoencoder()
        classifier = SscnRotationClassifier(num_clusters=num_clusters,
                                            max_cluster_dim=max_cluster_dim,
                                            )

        super().__init__(encoder=encoder,
                         decoder=decoder,
                         classifier=classifier,
                         hidden_size=num_features,
                         num_clusters=num_clusters,
                         max_cluster_dim=max_cluster_dim,
                         lambda_reconst=0,
                         lambda_reg=lambda_reg,
                         lambda_ssc=lambda_ssc,
                         lambda_assignment=lambda_assignment,
                         se_bias=se_bias,
                         )


class SscnConv(Sscn):
    def __init__(self,
                 image_size: Tuple[int, int],
                 in_channels: int,
                 kernel_size: List[int],
                 num_channels: List[int],
                 stride: int,
                 padding: List[int],
                 num_clusters: int,
                 max_cluster_dim: int,
                 lambda_reconst: float,
                 lambda_reg: float,
                 lambda_ssc: float,
                 lambda_assignment: float,
                 se_bias: bool,
                 ):
        encoder, decoder, hidden_size = get_conv_autoencoder(image_size=image_size,
                                                             in_channels=in_channels,
                                                             kernel_size=kernel_size,
                                                             num_channels=num_channels,
                                                             stride=stride,
                                                             padding=padding,
                                                             )

        classifier = SscnRotationClassifier(num_clusters=num_clusters,
                                            max_cluster_dim=max_cluster_dim,
                                            )

        super().__init__(encoder=encoder,
                         decoder=decoder,
                         classifier=classifier,
                         hidden_size=hidden_size,
                         num_clusters=num_clusters,
                         max_cluster_dim=max_cluster_dim,
                         lambda_reconst=lambda_reconst,
                         lambda_reg=lambda_reg,
                         lambda_ssc=lambda_ssc,
                         lambda_assignment=lambda_assignment,
                         se_bias=se_bias,
                         )


class SscnClassifier(Module,
                     metaclass=abc.ABCMeta,
                     ):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_outputs(self,
                    H: Tensor,
                    ) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_soft_assignments(self,
                             H: Tensor,
                             ) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_optimizer(self,
                      learning_rate: float,
                      ) -> Optimizer:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset_parameters(self):
        raise NotImplementedError()


class SscnRotationClassifier(SscnClassifier):
    def __init__(self,
                 num_clusters: int,
                 max_cluster_dim: int,
                 ):
        super().__init__()

        hidden_size_se = num_clusters * max_cluster_dim

        # Subspace bases
        S = [torch.zeros((hidden_size_se, max_cluster_dim))
             for _ in range(num_clusters)]
        start = 0
        for i in range(num_clusters):
            S[i][range(start, start + max_cluster_dim), range(max_cluster_dim)] = 1.
            start += max_cluster_dim
        S = torch.stack(S, axis=0)
        self.S = Parameter(S,
                           requires_grad=False,
                           )

        # Rotation matrix
        self.R = Parameter(torch.eye(hidden_size_se),
                           requires_grad=True,
                           )

    def get_rotated_embeddings(self,
                               H: Tensor,
                               ) -> Tensor:
        return H @ self.R

    def get_outputs(self,
                    H: Tensor,
                    ) -> Tensor:
        H_proj = self.get_rotated_embeddings(H)
        projection_distance = get_projection_distance(H_proj, self.S)

        return -projection_distance

    def get_soft_assignments(self,
                             H: Tensor,
                             ) -> Tensor:
        outputs = self.get_outputs(H)
        soft_assignments = torch.nn.functional.softmax(outputs, dim=-1)

        return soft_assignments

    def get_optimizer(self,
                      learning_rate: float,
                      ) -> Optimizer:
        dict_stiefel = {'params': self.R,
                        'lr': learning_rate,
                        'momentum': 0.9,
                        'stiefel': True,
                        }
        optimizer = AdamG([dict_stiefel])

        return optimizer

    def reset_parameters(self):
        eye_(self.R)
