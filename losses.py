from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class NTXentLossSupervised(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLossSupervised, self).__init__()
        self.temperature = temperature

    def forward(self, out0, out1, labels):
        out0 = out0[labels != -1]
        out1 = out1[labels != -1]
        b = out0.shape[0]
        z1 = torch.nn.functional.normalize(out0)
        z2 = torch.nn.functional.normalize(out1)
        z11 = z1 @ z1.T / self.temperature
        z22 = z2 @ z2.T / self.temperature
        z12 = z1 @ z2.T / self.temperature

        l = labels[labels != -1]
        mij = l.unsqueeze(0) == l.unsqueeze(1)
        mii = l.unsqueeze(0) == l.unsqueeze(1)
        mii.view(-1)[:: (b + 1)].fill_(bool(0))

        p1 = (
            torch.hstack((z11 * mii.float(), z12 * mij.float())).sum(dim=1)
            / torch.hstack((mii, mij)).sum(dim=1).float()
        )
        p2 = (
            torch.hstack((z12.T * mij.float(), z22 * mii.float())).sum(dim=1)
            / torch.hstack((mij, mii)).sum(dim=1).float()
        )
        pos = torch.cat((p1, p2)).mean()

        z11.view(-1)[:: (b + 1)].fill_(float("-inf"))
        z22.view(-1)[:: (b + 1)].fill_(float("-inf"))

        n1 = torch.hstack((z11, z12)).logsumexp(dim=1)
        n2 = torch.hstack((z12.T, z22)).logsumexp(dim=1)
        neg = torch.cat((n1, n2)).mean()
        loss = -(pos - neg)
        return loss


class CombinedLoss(torch.nn.Module):
    def __init__(self, loss1, loss2, lambda_=0.35):
        super(CombinedLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.lambda_ = lambda_

    def forward(self, out0, out1, labels):
        # if no label is not -1, then use loss1
        if (labels == -1).all():
            unsup_loss = self.loss1(out0, out1)
            sup_loss = 0.0
        else:
            unsup_loss = self.loss1(out0, out1)
            sup_loss = self.loss2(out0, out1, labels)
        return (1 - self.lambda_) * unsup_loss + self.lambda_ * sup_loss


# copied from lightly package - modifying to use labels
class DINOLoss(nn.Module):
    """
    Implementation of the loss described in 'Emerging Properties in
    Self-Supervised Vision Transformers'. [0]

    This implementation follows the code published by the authors. [1]
    It supports global and local image crops. A linear warmup schedule for the
    teacher temperature is implemented to stabilize training at the beginning.
    Centering is applied to the teacher output to avoid model collapse.

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: https://github.com/facebookresearch/dino

    Attributes:
        output_dim:
            Dimension of the model output.
        warmup_teacher_temp:
            Initial value of the teacher temperature. Should be decreased if the
            training loss does not decrease.
        teacher_temp:
            Final value of the teacher temperature after linear warmup. Values
            above 0.07 result in unstable behavior in most cases. Can be
            slightly increased to improve performance during finetuning.
        warmup_teacher_temp_epochs:
            Number of epochs for the teacher temperature warmup.
        student_temp:
            Temperature of the student.
        center_momentum:
            Momentum term for the center calculation.
    """

    def __init__(
        self,
        output_dim: int = 65536,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        target_dist: torch.Tensor = None,
        lambda0: float = 0.65,
        lambda1: float = 0.35,
        lambda2: float = 2.0,
    ):
        super().__init__()
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.register_buffer("center", torch.zeros(1, output_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = torch.linspace(
            start=warmup_teacher_temp,
            end=teacher_temp,
            steps=warmup_teacher_temp_epochs,
        )
        self.target_dist = target_dist

    def forward(
        self,
        teacher_out: List[torch.Tensor],
        student_out: List[torch.Tensor],
        labels: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """Cross-entropy between softmax outputs of the teacher and student
        networks.

        Args:
            teacher_out:
                List of view feature tensors from the teacher model. Each
                tensor is assumed to contain features from one view of the batch
                and have length batch_size.
            student_out:
                List of view feature tensors from the student model. Each tensor
                is assumed to contain features from one view of the batch and
                have length batch_size.
            epoch:
                The current training epoch.

        Returns:
            The average cross-entropy loss.

        """
        # get teacher temperature
        if epoch < self.warmup_teacher_temp_epochs:
            teacher_temp = self.teacher_temp_schedule[epoch]
        else:
            teacher_temp = self.teacher_temp

        t_out = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1)

        s_out = F.log_softmax(student_out / self.student_temp, dim=-1)

        # unsupervised loss is cross entropy between student and teacher
        loss = torch.mean(torch.sum(-t_out * s_out, dim=-1))

        # supervised loss is cross entropy between student and the labels
        lbl = labels[labels != -1]  # b
        if len(lbl) == 0:
            sup_loss = 0.0
        else:
            s_out = student_out[labels != -1]
            sup_loss = F.cross_entropy(s_out, lbl)

        if self.target_dist is not None:
            # regularize the student output to be similar to the target distribution
            mean_student_probs = torch.mean(F.softmax(student_out, dim=-1), dim=0)
            target_dist = self.target_dist.to(mean_student_probs.device)
            # kl divergence between the student output and the target distribution
            dist_loss = torch.sum(mean_student_probs * torch.log(mean_student_probs / target_dist))
        else:
            mean_student_probs = torch.mean(F.softmax(student_out, dim=-1), dim=0)
            # lets just minimize negative entropy of the student
            dist_loss = torch.sum(mean_student_probs * torch.log(mean_student_probs))
        # combine the losses
        total_loss = (
            self.lambda0 * loss + self.lambda1 * sup_loss + self.lambda2 * dist_loss
        )

        self.update_center(teacher_out)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_out: torch.Tensor) -> None:
        """Moving average update of the center used for the teacher output.

        Args:
            teacher_out:
                Stacked output from the teacher model.

        """
        batch_center = torch.mean(teacher_out, dim=(0), keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )
