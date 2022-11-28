import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D


class SmallDeepSet(nn.Module):
    def __init__(self, input_features, pool="mean", thres=0.5, reg=False):
        super().__init__()
        self.input_features = input_features
        self.enc = nn.Sequential(
            nn.Linear(in_features=self.input_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid(),
        )
        self.pool = pool
        self.thres = thres
        self.reg = reg

        self.reg_dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, x):
        x = self.enc(x)
        if self.pool == "max":
            x = x.max(dim=1)[0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        elif self.pool == "min":
            x = x.min(dim=1)[0]
        if self.reg:
            x = self.reg_dec(x)
        else:
            x = self.dec(x)
        return x  # , torch.ge(x, self.thres)


class profile_AttSet(nn.Module):
    """
    This implements the DeepAttentionMIL model https://github.com/AMLab-Amsterdam/AttentionDeepMIL

    

    """
    def __init__(self, input_feature, pool="att", thres=0.5, reg=False):
        super(profile_AttSet, self).__init__()

        self.input_feature = input_feature
        self.pool = pool
        self.L = 80  # 230
        self.D = 36  # 128
        self.K = 1
        self.thres = thres
        self.reg = reg
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.input_feature, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1), nn.Sigmoid())
        self.regressor = nn.Sequential(nn.Linear(self.L * self.K, 1))

    def forward(self, x):
        # H = x.squeeze(0)
        H = self.feature_extractor_part2(x)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 2, 1)  # KxN
        A = F.softmax(A, dim=2)  # softmax over N
        M = torch.bmm(A, H)  # KxL
        if self.reg:
            y = self.regressor(M)
            return y.view(-1, 1)
        else:
            Y_prob = self.classifier(M)
            Y_prob = Y_prob.squeeze(2)
            Y_hat = torch.ge(Y_prob, self.thres).float()
            return Y_prob, Y_hat  # , A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.0 - 1e-5)
        neg_log_likelihood = -1.0 * (
            Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob)
        )  # negative log bernoulli

        return neg_log_likelihood, A


class transformer(nn.Module):
    """
    This implements the SetTransformer model https://github.com/juho-lee/set_transformer

    """
    def __init__(self, input_feature, thres=0.5, reg=False):
        super(transformer, self).__init__()

        self.input_feature = input_feature
        self.reg = reg
        self.thres = thres
        self.reg = reg
        self.L = 80
        self.D = 36
        self.K = 1

        self.attention = nn.Transformer(
            d_model = input_feature,
            nhead = 4,
            num_encoder_layers = 3,
            num_decoder_layers = 1,
            dim_feedforward = 256,
            batch_first = True)
        self.pma = nn.Transformer(
            d_model = input_feature,
            nhead = 1,
            num_encoder_layers = 1,
            num_decoder_layers = 1,
            dim_feedforward = 64,
            batch_first = True)

        self.S = nn.Parameter(torch.Tensor(1, 1, self.input_feature))
        nn.init.xavier_uniform_(self.S)

        if self.reg:
            self.regressor = nn.Sequential(nn.Linear(input_feature * self.K, 1))
        else:
            self.classifier = nn.Sequential(nn.Linear(input_feature * self.K, 1), nn.Sigmoid())

    def forward(self, x):
        M = self.attention(x, x)
        pooled = self.pma(M, self.S.repeat(M.size(0), self.K, self.K))
        if self.reg:
            output = self.regressor(pooled)
            return output.view(-1, 1)
        else:
            Y_prob = self.classifier(pooled)
            Y_prob = Y_prob.squeeze(2)
            Y_hat = torch.ge(Y_prob, self.thres).float()
            return Y_prob, Y_hat  # , A            


    
class simpling_pooling(SmallDeepSet):
    def __init__(self, input_features, pool="mean", thres=0.5, reg=False):
        super().__init__(input_features, pool, thres, reg)
        self.reg_dec = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=1),
        )

    def forward(self, x):
        if self.pool == "max":
            x = x.max(dim=1)[0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        elif self.pool == "min":
            x = x.min(dim=1)[0]
        if self.reg:
            x = self.reg_dec(x)
        else:
            x = self.dec(x)
        return x
