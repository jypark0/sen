import e2cnn.nn as e2nn
import torch.nn as nn
from e2cnn import gspaces

from sen import utils


class CubesEmbedding(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        act_fn="sigmoid",
        act_fn_hid="relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.act2 = utils.get_act_fn(act_fn_hid)
        self.ln2 = nn.BatchNorm2d(hidden_dim)

        self.cnn3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.act3 = utils.get_act_fn(act_fn_hid)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.cnn4 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.act4 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.ln2(self.cnn2(h)))
        h = self.act3(self.ln3(self.cnn3(h)))
        return self.act4(self.cnn4(h))


class CubesEmbedding_E2(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, group_order, out_feat_type="trivial"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.r2_act = gspaces.Rot2dOnR2(N=group_order)

        self.feat_type_in = e2nn.FieldType(
            self.r2_act, input_dim * [self.r2_act.trivial_repr]
        )
        self.feat_type_hid = e2nn.FieldType(
            self.r2_act, hidden_dim * [self.r2_act.regular_repr]
        )
        self.out_feat_type = out_feat_type
        if out_feat_type == "trivial":
            self.feat_type_out = e2nn.FieldType(
                self.r2_act, output_dim * [self.r2_act.trivial_repr]
            )
        elif out_feat_type == "regular":
            self.feat_type_out = e2nn.FieldType(
                self.r2_act, output_dim * [self.r2_act.regular_repr]
            )
        else:
            raise NotImplementedError

        self.net = e2nn.SequentialModule(
            e2nn.R2Conv(self.feat_type_in, self.feat_type_hid, 3, padding=1),
            e2nn.InnerBatchNorm(self.feat_type_hid),
            e2nn.ReLU(self.feat_type_hid),
            e2nn.R2Conv(self.feat_type_hid, self.feat_type_hid, 3, padding=1),
            e2nn.InnerBatchNorm(self.feat_type_hid),
            e2nn.ReLU(self.feat_type_hid),
            e2nn.R2Conv(self.feat_type_hid, self.feat_type_hid, 3, padding=1),
            e2nn.InnerBatchNorm(self.feat_type_hid),
            e2nn.ReLU(self.feat_type_hid),
            e2nn.R2Conv(self.feat_type_hid, self.feat_type_out, 3, padding=1),
            e2nn.PointwiseNonLinearity(self.feat_type_out, function="p_sigmoid"),
        )

        # [B, O*4, 50, 50] -> [B, O, 4, 50, 50]
        self.last = nn.Unflatten(1, (-1, group_order))

    def forward(self, x):
        x = e2nn.GeometricTensor(x, self.feat_type_in)
        x = self.net(x)
        if self.out_feat_type == "regular":
            x = self.last(x.tensor)
        else:
            x = x.tensor

        return x


class ReacherEmbedding(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=[16, 16, 16, 16, 16, 16],
        act_fn_hid="relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        assert len(hidden_dim) == 6, "len(hidden_dim) must equal 6"
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim[0], 3, stride=2, dilation=1, padding=1),
            nn.BatchNorm2d(hidden_dim[0]),
            utils.get_act_fn(act_fn_hid),
            nn.Conv2d(hidden_dim[0], hidden_dim[1], 3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(hidden_dim[1]),
            utils.get_act_fn(act_fn_hid),
            nn.Conv2d(hidden_dim[1], hidden_dim[2], 3, stride=1, dilation=4, padding=4),
            nn.BatchNorm2d(hidden_dim[2]),
            utils.get_act_fn(act_fn_hid),
            nn.Conv2d(hidden_dim[2], hidden_dim[3], 3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(hidden_dim[3]),
            utils.get_act_fn(act_fn_hid),
            nn.Conv2d(
                hidden_dim[3], hidden_dim[4], 3, stride=1, dilation=16, padding=16
            ),
            nn.BatchNorm2d(hidden_dim[4]),
            utils.get_act_fn(act_fn_hid),
            nn.Conv2d(
                hidden_dim[4], hidden_dim[5], 3, stride=1, dilation=32, padding=32
            ),
            nn.BatchNorm2d(hidden_dim[5]),
            utils.get_act_fn(act_fn_hid),
            nn.Conv2d(hidden_dim[5], output_dim, 3, stride=2, dilation=1, padding=1),
        )

    def forward(self, obs):
        return self.net(obs)


class ReacherEmbedding_D4(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        group_order,
        hidden_dim=[16, 16, 16, 16, 16, 16],
    ):
        super().__init__()
        self.input_dim = input_dim
        assert len(hidden_dim) == 6, "len(hidden_dim) must equal 6"
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.r2_act = gspaces.FlipRot2dOnR2(N=group_order)

        self.feat_type_in = e2nn.FieldType(
            self.r2_act, input_dim * [self.r2_act.trivial_repr]
        )
        self.feat_type_hid = [
            e2nn.FieldType(self.r2_act, h * [self.r2_act.regular_repr])
            for h in hidden_dim
        ]
        self.feat_type_out = e2nn.FieldType(
            self.r2_act, output_dim * [self.r2_act.regular_repr]
        )

        self.net = e2nn.SequentialModule(
            e2nn.R2Conv(
                self.feat_type_in, self.feat_type_hid[0], 3, stride=2, padding=1
            ),
            e2nn.InnerBatchNorm(self.feat_type_hid[0]),
            e2nn.ReLU(self.feat_type_hid[0]),
            e2nn.R2Conv(
                self.feat_type_hid[0], self.feat_type_hid[1], 3, dilation=2, padding=2
            ),
            e2nn.InnerBatchNorm(self.feat_type_hid[1]),
            e2nn.ReLU(self.feat_type_hid[1]),
            e2nn.R2Conv(
                self.feat_type_hid[1], self.feat_type_hid[2], 3, dilation=4, padding=4
            ),
            e2nn.InnerBatchNorm(self.feat_type_hid[2]),
            e2nn.ReLU(self.feat_type_hid[2]),
            e2nn.R2Conv(
                self.feat_type_hid[2], self.feat_type_hid[3], 3, dilation=8, padding=8
            ),
            e2nn.InnerBatchNorm(self.feat_type_hid[3]),
            e2nn.ReLU(self.feat_type_hid[3]),
            e2nn.R2Conv(
                self.feat_type_hid[3], self.feat_type_hid[4], 3, dilation=16, padding=16
            ),
            e2nn.InnerBatchNorm(self.feat_type_hid[4]),
            e2nn.ReLU(self.feat_type_hid[4]),
            e2nn.R2Conv(
                self.feat_type_hid[4], self.feat_type_hid[5], 3, dilation=32, padding=32
            ),
            e2nn.InnerBatchNorm(self.feat_type_hid[5]),
            e2nn.ReLU(self.feat_type_hid[5]),
            e2nn.R2Conv(
                self.feat_type_hid[5], self.feat_type_out, 3, stride=2, padding=1
            ),
        )

    def forward(self, obs):
        obs = e2nn.GeometricTensor(obs, self.feat_type_in)
        x = self.net(obs)

        return x.tensor
