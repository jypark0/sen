import numpy as np
import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as e2nn

from sen import utils
from sen.net.layers import C4Conv


class ReacherEncoder_D4(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        group_order,
        hidden_dim=[256, 256, 256, 256, 256],
        in_feat_type="trivial",
    ):
        super().__init__()
        self.input_dim = input_dim
        assert len(hidden_dim) == 5, "len(hidden_dim) must equal 5"
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.r2_act = gspaces.FlipRot2dOnR2(N=group_order)

        if in_feat_type == "trivial":
            self.feat_type_in = e2nn.FieldType(
                self.r2_act, input_dim * [self.r2_act.trivial_repr]
            )
        elif in_feat_type == "regular":
            self.feat_type_in = e2nn.FieldType(
                self.r2_act, input_dim * [self.r2_act.regular_repr]
            )
        else:
            raise NotImplementedError

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
            e2nn.PointwiseMaxPool(self.feat_type_hid[0], 2),
            e2nn.R2Conv(
                self.feat_type_hid[0], self.feat_type_hid[1], 3, stride=2, padding=1
            ),
            e2nn.InnerBatchNorm(self.feat_type_hid[1]),
            e2nn.ReLU(self.feat_type_hid[1]),
            e2nn.PointwiseMaxPool(self.feat_type_hid[1], 2),
            e2nn.R2Conv(
                self.feat_type_hid[1], self.feat_type_hid[2], 3, stride=1, padding=1
            ),
            e2nn.InnerBatchNorm(self.feat_type_hid[2]),
            e2nn.ReLU(self.feat_type_hid[2]),
            e2nn.PointwiseMaxPool(self.feat_type_hid[2], 2),
            e2nn.R2Conv(
                self.feat_type_hid[2], self.feat_type_hid[3], 1, stride=1, padding=0
            ),
            e2nn.InnerBatchNorm(self.feat_type_hid[3]),
            e2nn.ReLU(self.feat_type_hid[3]),
            e2nn.R2Conv(
                self.feat_type_hid[3], self.feat_type_hid[4], 1, stride=1, padding=0
            ),
            e2nn.InnerBatchNorm(self.feat_type_hid[4]),
            e2nn.ReLU(self.feat_type_hid[4]),
            e2nn.R2Conv(
                self.feat_type_hid[4], self.feat_type_out, 1, stride=1, padding=0
            ),
        )

        # Shape: [B, obs_dim*2*group_order, 1, 1] ->  [B, obs_dim, 2*group_order]
        self.last = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Unflatten(dim=-1, unflattened_size=(output_dim, 2 * group_order)),
        )

    def forward(self, x):
        x = e2nn.GeometricTensor(x, self.feat_type_in)
        y = self.net(x)

        # Shape back into [B, output_dim, group_order]
        y = self.last(y.tensor)
        return y


class ReacherEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[64, 128, 256, 512, 512]):

        super().__init__()
        self.input_dim = input_dim
        assert len(hidden_dim) == 5, "len(hidden_dim) must equal 5"
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim[0], hidden_dim[1], 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim[1], hidden_dim[2], 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim[2]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim[2], hidden_dim[3], 1, padding=0),
            nn.BatchNorm2d(hidden_dim[3]),
            nn.ReLU(),
            nn.Conv2d(hidden_dim[3], hidden_dim[4], 1, padding=0),
            nn.BatchNorm2d(hidden_dim[4]),
            nn.ReLU(),
            nn.Conv2d(hidden_dim[4], output_dim, 1, padding=0),
        )
        self.last = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        y = self.net(x)

        # Squeeze last 1x1 dimensions
        y = self.last(y)
        return y


class CubesEncoder(nn.Module):
    """MLP encoder, maps observation to latent state."""

    def __init__(
        self,
        input_shape,
        output_shape,
        num_objects,
        hidden_dim=[256, 256],
        act_fn_hid="relu",
    ):
        super().__init__()
        self.input_shape = input_shape
        assert len(hidden_dim) == 2, "len(hidden_dim) must equal 2"
        self.hidden_dim = hidden_dim
        self.output_shape = output_shape
        self.num_objects = num_objects

        # [B,O,32,32] -> [B,O,32*32]
        self.first = nn.Flatten(start_dim=2, end_dim=-1)

        self.net = nn.Sequential(
            nn.Linear(np.prod(input_shape[1:]), hidden_dim[0]),
            utils.get_act_fn(act_fn_hid),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.LayerNorm(hidden_dim[1]),
            utils.get_act_fn(act_fn_hid),
            nn.Linear(hidden_dim[1], np.prod(output_shape)),
        )
        if isinstance(output_shape, int):
            self.last = nn.Identity()
        elif isinstance(output_shape, list):
            self.last = nn.Unflatten(-1, output_shape)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.first(x)
        x = self.net(x)
        x = self.last(x)
        return x


class CubesEncoder_C4(nn.Module):
    def __init__(
        self, input_shape, output_dim, num_objects, hidden_dim=[384, 384], act_fn="relu"
    ):

        super().__init__()
        if input_shape[1] % 2 == 0:
            # even
            in_dim = (input_shape[1] // 2) * (input_shape[2] // 2)
        else:
            # odd
            in_dim = ((input_shape[1] - 1) // 2) * ((input_shape[2] - 1) // 2 + 1) + 1

        self.input_dim = in_dim
        assert len(hidden_dim) == 2, "len(hidden_dim) must equal 2"
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_objects = num_objects

        self.fc1 = C4Conv(self.input_dim, hidden_dim[0])
        self.fc2 = C4Conv(hidden_dim[0], hidden_dim[1])
        self.fc3 = C4Conv(hidden_dim[1], output_dim)
        # input [batch, num_objects, 4, hidden_dim], normalize over last two dims
        self.ln = nn.LayerNorm(hidden_dim[1])

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        """
        Input shape: [B, O, H, H]
        """
        # [batch, num_objects, height (e.g. 5), width (e.g. 5)]
        h_flat = self.orbit_stack(ins)
        # [batch, num_objects, 4, number of orbits / channel dimension (e.g. 7)]
        h = self.act1(self.fc1(h_flat))
        # [batch, num_objects, 4, hidden_dim]
        h = self.act2(self.ln(self.fc2(h)))
        # [batch, num_objects, 4, output_dim]
        return self.fc3(h)

    def orbit_stack(self, x):
        """
        Input: (batch, num_objects, H, H)
        """
        H = x.size(-1)

        if H % 2 == 1:
            # odd
            c = (H - 1) // 2
            n_orbits = 1 + c + (c) ** 2

            out = torch.zeros(
                (x.size(0), x.size(1), n_orbits, 4), device=self.fc1.weights.device
            )

            out[:, :, 0, :] = ((x[:, :, c, c]).unsqueeze(2)).expand(-1, -1, 4)
            for i in range(4):
                out[:, :, 1:, i] = torch.rot90(x, i, (-2, -1))[
                    :, :, :c, : c + 1
                ].reshape(x.size(0), x.size(1), -1)
        else:
            # even
            n_orbits = (H // 2) ** 2
            c = H // 2

            out = torch.zeros(
                (x.size(0), x.size(1), n_orbits, 4), device=self.fc1.weights.device
            )

            for i in range(4):
                out[:, :, :, i] = torch.rot90(x, i, (-2, -1))[:, :, :c, :c].reshape(
                    x.size(0), x.size(1), -1
                )

        return out.permute(0, 1, 3, 2)
