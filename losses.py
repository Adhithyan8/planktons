import torch
import torch.distributed as dist


class InfoNCECosineSelfSupervised(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCECosineSelfSupervised, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        b = features.shape[0] // 2
        z1 = torch.nn.functional.normalize(features[:b])
        z2 = torch.nn.functional.normalize(features[b:])
        z11 = z1 @ z1.T / self.temperature
        z22 = z2 @ z2.T / self.temperature
        z12 = z1 @ z2.T / self.temperature

        pos = torch.trace(z12) / b

        n1 = (
            torch.hstack(
                (
                    z11.masked_fill_(torch.eye(b).to(z11).bool(), float("-inf")),
                    z12,
                )
            )
            .logsumexp(dim=1)
            .mean()
        )
        n2 = (
            torch.hstack(
                (
                    z12.T,
                    z22.masked_fill_(torch.eye(b).to(z22).bool(), float("-inf")),
                )
            )
            .logsumexp(dim=1)
            .mean()
        )
        neg = (n1 + n2) / 2
        loss = -(pos - neg)
        return loss


class InfoNCECosineSupervised(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCECosineSupervised, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        f = features[labels != -1]
        b = f.shape[0] // 2
        z1 = torch.nn.functional.normalize(f[:b])
        z2 = torch.nn.functional.normalize(f[b:])
        z11 = z1 @ z1.T / self.temperature
        z22 = z2 @ z2.T / self.temperature
        z12 = z1 @ z2.T / self.temperature

        l = labels[labels != -1][:b]
        mij = l.unsqueeze(0) == l.unsqueeze(1)
        mii = (l.unsqueeze(0) == l.unsqueeze(1)).masked_fill_(
            torch.eye(b).to(mij), bool(0)
        )

        p1 = (
            torch.hstack((z11 * mii.float(), z12 * mij.float())).sum(dim=1)
            / torch.hstack((mii, mij)).sum(dim=1).float()
        )
        p2 = (
            torch.hstack((z12.T * mij.float(), z22 * mii.float())).sum(dim=1)
            / torch.hstack((mij, mii)).sum(dim=1).float()
        )
        pos = torch.cat((p1, p2)).mean()

        n1 = (
            torch.hstack(
                (
                    z11.masked_fill_(torch.eye(b).to(mij), float("-inf")),
                    z12,
                )
            )
            .logsumexp(dim=1)
            .mean()
        )
        n2 = (
            torch.hstack(
                (
                    z12.T,
                    z22.masked_fill_(torch.eye(b).to(mij), float("-inf")),
                )
            )
            .logsumexp(dim=1)
            .mean()
        )
        neg = (n1 + n2) / 2
        loss = -(pos - neg)
        return loss


class InfoNCECosineSemiSupervised(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCECosineSemiSupervised, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        b = features.shape[0] // 2
        l = labels[:b]
        l_l = l[l != -1]
        b_l = l_l.shape[0]
        b_u = b - b_l
        z_u1 = torch.nn.functional.normalize(features[:b][l == -1])
        z_u2 = torch.nn.functional.normalize(features[b:][l == -1])
        z_l1 = torch.nn.functional.normalize(features[:b][l != -1])
        z_l2 = torch.nn.functional.normalize(features[b:][l != -1])

        z_u11 = z_u1 @ z_u1.T / self.temperature
        z_u22 = z_u2 @ z_u2.T / self.temperature
        z_u12 = z_u1 @ z_u2.T / self.temperature
        p1 = torch.trace(z_u12) / b

        z_l11 = z_l1 @ z_l1.T / self.temperature
        z_l22 = z_l2 @ z_l2.T / self.temperature
        z_l12 = z_l1 @ z_l2.T / self.temperature
        m_ij = l_l.unsqueeze(0) == l_l.unsqueeze(1)
        m_ii = (l_l.unsqueeze(0) == l_l.unsqueeze(1)).masked_fill_(
            torch.eye(l_l.shape[0]).to(m_ij), bool(0)
        )
        p2 = torch.hstack((z_l11 * m_ii.float(), z_l12 * m_ij.float())).sum(
            dim=1
        ) / torch.hstack((m_ii, m_ij)).sum(dim=1)
        p3 = torch.hstack((z_l12.T * m_ij.float(), z_l22 * m_ii.float())).sum(
            dim=1
        ) / torch.hstack((m_ij, m_ii)).sum(dim=1)
        pos = torch.cat((p2, p3)).sum().div(2 * b) + p1

        z_u1l1 = z_u1 @ z_l1.T / self.temperature
        z_u1l2 = z_u1 @ z_l2.T / self.temperature
        z_u2l1 = z_u2 @ z_l1.T / self.temperature
        z_u2l2 = z_u2 @ z_l2.T / self.temperature
        n1 = torch.hstack(
            (
                z_u11.masked_fill_(torch.eye(b_u).to(m_ij), float("-inf")),
                z_u12,
                z_u1l1,
                z_u1l2,
            )
        ).logsumexp(dim=1)
        n2 = torch.hstack(
            (
                z_u12.T,
                z_u22.masked_fill_(torch.eye(b_u).to(m_ij), float("-inf")),
                z_u2l1,
                z_u2l2,
            )
        ).logsumexp(dim=1)
        n3 = torch.hstack(
            (
                z_u1l1.T,
                z_u2l1.T,
                z_l11.masked_fill_(torch.eye(b_l).to(m_ij), float("-inf")),
                z_l12,
            )
        ).logsumexp(dim=1)
        n4 = torch.hstack(
            (
                z_u1l2.T,
                z_u2l2.T,
                z_l12.T,
                z_l22.masked_fill_(torch.eye(b_l).to(m_ij), float("-inf")),
            )
        ).logsumexp(dim=1)
        neg = torch.cat((n1, n2, n3, n4)).mean()

        loss = -(pos - neg)
        return loss


class InfoNCECauchySelfSupervised(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(InfoNCECauchySelfSupervised, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        b = features.shape[0] // 2
        z1 = features[:b]
        z2 = features[b:]
        z11 = 1.0 / (
            1.0
            + (z1.unsqueeze(1) - z1.unsqueeze(0)).square().sum(dim=2) / self.temperature
        )
        z22 = 1.0 / (
            1.0
            + (z2.unsqueeze(1) - z2.unsqueeze(0)).square().sum(dim=2) / self.temperature
        )
        z12 = 1.0 / (
            1.0
            + (z1.unsqueeze(1) - z2.unsqueeze(0)).square().sum(dim=2) / self.temperature
        )

        pos = torch.trace(z12.log()) / b

        n1 = torch.hstack((z11 - torch.eye(b).to(z11), z12)).sum(dim=1).log().mean()
        n2 = torch.hstack((z12.T, z22 - torch.eye(b).to(z22))).sum(dim=1).log().mean()
        neg = (n1 + n2) / 2
        loss = -(pos - neg)
        return loss


class InfoNCECauchySupervised(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(InfoNCECauchySupervised, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        f = features[labels != -1]
        b = f.shape[0] // 2
        z1 = f[:b]
        z2 = f[b:]
        z11 = 1.0 / (
            1.0
            + (z1.unsqueeze(1) - z1.unsqueeze(0)).square().sum(dim=2) / self.temperature
        )
        z22 = 1.0 / (
            1.0
            + (z2.unsqueeze(1) - z2.unsqueeze(0)).square().sum(dim=2) / self.temperature
        )
        z12 = 1.0 / (
            1.0
            + (z1.unsqueeze(1) - z2.unsqueeze(0)).square().sum(dim=2) / self.temperature
        )

        l = labels[labels != -1][:b]
        mij = l.unsqueeze(0) == l.unsqueeze(1)
        mii = (l.unsqueeze(0) == l.unsqueeze(1)).masked_fill_(
            torch.eye(b).to(mij), bool(0)
        )

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


class InfoNCECauchySemiSupervised(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(InfoNCECauchySemiSupervised, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        b = features.shape[0] // 2
        l = labels[:b]
        l_l = l[l != -1]
        b_l = l_l.shape[0]
        b_u = b - b_l
        z_u1 = features[:b][l == -1]
        z_u2 = features[b:][l == -1]
        z_l1 = features[:b][l != -1]
        z_l2 = features[b:][l != -1]

        z_u11 = 1.0 / (
            1.0
            + (z_u1.unsqueeze(1) - z_u1.unsqueeze(0)).square().sum(dim=2)
            / self.temperature
        )
        z_u22 = 1.0 / (
            1.0
            + (z_u2.unsqueeze(1) - z_u2.unsqueeze(0)).square().sum(dim=2)
            / self.temperature
        )
        z_u12 = 1.0 / (
            1.0
            + (z_u1.unsqueeze(1) - z_u2.unsqueeze(0)).square().sum(dim=2)
            / self.temperature
        )
        p1 = torch.trace(z_u12.log()) / b

        z_l11 = 1.0 / (
            1.0
            + (z_l1.unsqueeze(1) - z_l1.unsqueeze(0)).square().sum(dim=2)
            / self.temperature
        )
        z_l22 = 1.0 / (
            1.0
            + (z_l2.unsqueeze(1) - z_l2.unsqueeze(0)).square().sum(dim=2)
            / self.temperature
        )
        z_l12 = 1.0 / (
            1.0
            + (z_l1.unsqueeze(1) - z_l2.unsqueeze(0)).square().sum(dim=2)
            / self.temperature
        )
        m_ij = l_l.unsqueeze(0) == l_l.unsqueeze(1)
        m_ii = (l_l.unsqueeze(0) == l_l.unsqueeze(1)).masked_fill_(
            torch.eye(l_l.shape[0]).to(m_ij), bool(0)
        )
        p2 = (
            torch.hstack((z_l11.log() * m_ii.float(), z_l12.log() * m_ij.float())).sum(
                dim=1
            )
            / torch.hstack((m_ii, m_ij)).sum(dim=1).float()
        )
        p3 = (
            torch.hstack((z_l12.log() * m_ij.float(), z_l22.log() * m_ii.float())).sum(
                dim=1
            )
            / torch.hstack((m_ij, m_ii)).sum(dim=1).float()
        )
        pos = torch.cat((p2, p3)).sum().div(2 * b) + p1

        z_u1l1 = 1.0 / (
            1.0
            + (z_u1.unsqueeze(1) - z_l1.unsqueeze(0)).square().sum(dim=2)
            / self.temperature
        )
        z_u1l2 = 1.0 / (
            1.0
            + (z_u1.unsqueeze(1) - z_l2.unsqueeze(0)).square().sum(dim=2)
            / self.temperature
        )
        z_u2l1 = 1.0 / (
            1.0
            + (z_u2.unsqueeze(1) - z_l1.unsqueeze(0)).square().sum(dim=2)
            / self.temperature
        )
        z_u2l2 = 1.0 / (
            1.0
            + (z_u2.unsqueeze(1) - z_l2.unsqueeze(0)).square().sum(dim=2)
            / self.temperature
        )
        n1 = (
            torch.hstack(
                (
                    z_u11 - torch.eye(b_u).to(z_u11),
                    z_u12,
                    z_u1l1,
                    z_u1l2,
                )
            )
            .sum(dim=1)
            .log()
        )
        n2 = (
            torch.hstack(
                (
                    z_u12.T,
                    z_u22 - torch.eye(b_u).to(z_u22),
                    z_u2l1,
                    z_u2l2,
                )
            )
            .sum(dim=1)
            .log()
        )
        n3 = (
            torch.hstack(
                (
                    z_u1l1.T,
                    z_u2l1.T,
                    z_l11 - torch.eye(b_l).to(z_l11),
                    z_l12,
                )
            )
            .sum(dim=1)
            .log()
        )
        n4 = (
            torch.hstack(
                (
                    z_u1l2.T,
                    z_u2l2.T,
                    z_l12.T,
                    z_l22 - torch.eye(b_l).to(z_l22),
                )
            )
            .sum(dim=1)
            .log()
        )
        neg = torch.cat((n1, n2, n3, n4)).mean()

        loss = -(pos - neg)
        return loss


class CombinedLoss(torch.nn.Module):
    def __init__(self, loss1, loss2, lambda_=1.0):
        super(CombinedLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.lambda_ = lambda_

    def forward(self, features, labels):
        return (1 - self.lambda_) * self.loss1(
            features, labels
        ) + self.lambda_ * self.loss2(features, labels)


class DistillLoss(torch.nn.Module):
    def __init__(
        self,
        epochs_warmup,
        epochs,
        teacher_temp_init=0.07,
        teacher_temp=0.04,
        student_temp=0.1,
        lambda_=0.35,
        lambda_reg=2.0,
        center_momentum=0.9,
        out_dim=128,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp_schedule = torch.cat(
            (
                torch.linspace(teacher_temp_init, teacher_temp, epochs_warmup),
                torch.ones(epochs - epochs_warmup) * teacher_temp,
            )
        )
        self.lambda_ = lambda_
        self.lambda_reg = lambda_reg
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, teacher_out, student_out, id, epoch):
        teacher_temp = self.teacher_temp_schedule[epoch]
        teacher_out = torch.nn.functional.softmax((teacher_out - self.center) / teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()
        student_out /= self.student_temp

        unsup_loss = torch.mean(
            -torch.sum(
                teacher_out * torch.nn.functional.log_softmax(student_out, dim=-1),
                dim=-1,
            )
        )
        self.update_center(teacher_out)

        label = id[id != -1]
        student_out_labeled = student_out[id != -1]
        if len(label) == 0:
            sup_loss = 0.0
        else:
            sup_loss = torch.nn.functional.cross_entropy(student_out_labeled, label)

        mean_student_probs = torch.mean(
            torch.nn.functional.softmax(student_out, dim=-1), dim=0
        )
        reg = torch.sum(mean_student_probs * torch.log(mean_student_probs))

        loss = (
            (1 - self.lambda_) * unsup_loss
            + self.lambda_ * sup_loss
            + self.lambda_reg * reg
        )
        return loss
    
    @torch.no_grad()
    def update_center(self, teacher_out):
        batch_center = torch.sum(teacher_out, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center /= dist.get_world_size()

        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class KoLeoLoss(torch.nn.Module):
    """
    Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search
    Reference: https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py
    """

    def __init__(self):
        super().__init__()
        self.pdist = torch.nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
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
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss


class DistillLoss2(torch.nn.Module):
    def __init__(
        self,
        lambda_=0.35,
        lambda_reg=0.0,
    ):
        super().__init__()
        self.lambda_ = lambda_
        self.lambda_reg = lambda_reg

    def forward(self, teacher_out, student_out, id, student_head_weights):
        teacher_out = teacher_out.detach()

        teacher_max = teacher_out.max(1)[1]
        unsup_loss = 1 - student_out[torch.arange(len(teacher_max)), teacher_max].mean()

        label = id[id != -1]
        student_out_labeled = student_out[id != -1]
        if len(label) == 0:
            sup_loss = 0.0
        else:
            sup_loss = 1 - student_out_labeled[torch.arange(len(label)), label].mean()

        reg = KoLeoLoss()(student_head_weights)

        loss = (
            (1 - self.lambda_) * unsup_loss
            + self.lambda_ * sup_loss
            + self.lambda_reg * reg
        )
        return loss
