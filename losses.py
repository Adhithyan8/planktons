import torch


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
