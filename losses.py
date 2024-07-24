import torch
import torch.distributed as dist


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

        n1 = torch.hstack((z11, z12)).logsumexp(dim=1).mean()
        n2 = torch.hstack((z12.T, z22)).logsumexp(dim=1).mean()
        neg = (n1 + n2) / 2
        loss = -(pos - neg)
        return loss


class InfoNCECauchy(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCECauchy, self).__init__()
        self.temperature = temperature

    def forward(self, out0, out1):
        b = out0.shape[0]
        z11 = 1.0 / (
            1.0
            + (out0.unsqueeze(1) - out0.unsqueeze(0)).square().sum(dim=2)
            / self.temperature
        )
        z22 = 1.0 / (
            1.0
            + (out1.unsqueeze(1) - out1.unsqueeze(0)).square().sum(dim=2)
            / self.temperature
        )
        z12 = 1.0 / (
            1.0
            + (out0.unsqueeze(1) - out1.unsqueeze(0)).square().sum(dim=2)
            / self.temperature
        )

        pos = torch.trace(z12.log()) / b

        n1 = torch.hstack((z11 - torch.eye(b).to(z11), z12)).sum(dim=1).log().mean()
        n2 = torch.hstack((z12.T, z22 - torch.eye(b).to(z22))).sum(dim=1).log().mean()
        neg = (n1 + n2) / 2
        loss = -(pos - neg)
        return loss


class InfoNCECauchySupervised(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCECauchySupervised, self).__init__()
        self.temperature = temperature

    def forward(self, out0, out1, labels):
        out0 = out0[labels != -1]
        out1 = out1[labels != -1]
        b = out0.shape[0]
        z11 = 1.0 / (
            1.0
            + (out0.unsqueeze(1) - out0.unsqueeze(0)).square().sum(dim=2)
            / self.temperature
        )
        z22 = 1.0 / (
            1.0
            + (out1.unsqueeze(1) - out1.unsqueeze(0)).square().sum(dim=2)
            / self.temperature
        )
        z12 = 1.0 / (
            1.0
            + (out0.unsqueeze(1) - out1.unsqueeze(0)).square().sum(dim=2)
            / self.temperature
        )

        l = labels[labels != -1]
        mij = l.unsqueeze(0) == l.unsqueeze(1)
        mii = l.unsqueeze(0) == l.unsqueeze(1)
        mii.view(-1)[:: (b + 1)].fill_(bool(0))

        p1 = (
            torch.hstack((z11.log() * mii.float(), z12.log() * mij.float())).sum(dim=1)
            / torch.hstack((mii, mij)).sum(dim=1).float()
        )
        p2 = (
            torch.hstack((z12.log() * mij.float(), z22.log() * mii.float())).sum(dim=1)
            / torch.hstack((mij, mii)).sum(dim=1).float()
        )
        pos = torch.cat((p1, p2)).mean()

        n1 = torch.hstack((z11 - torch.eye(b).to(z11), z12)).sum(dim=1).log().mean()
        n2 = torch.hstack((z12.T, z22 - torch.eye(b).to(z22))).sum(dim=1).log().mean()
        neg = (n1 + n2) / 2
        loss = -(pos - neg)
        return loss


class CombinedLoss(torch.nn.Module):
    def __init__(self, loss1, loss2, lambda_=0.35):
        super(CombinedLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.lambda_ = lambda_

    def forward(self, out0, out1, labels):
        return (1 - self.lambda_) * self.loss1(out0, out1) + self.lambda_ * self.loss2(
            out0, out1, labels
        )


class KoLeoLoss(torch.nn.Module):
    """
    Copied from https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py
    """

    def __init__(self):
        super().__init__()
        self.pdist = torch.nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):
            student_output = torch.nn.functional.normalize(
                student_output, eps=eps, p=2, dim=-1
            )
            I = self.pairwise_NNs_inner(student_output)
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss
