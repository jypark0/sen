import numpy as np
import torch
from torch import nn

from sen import utils


class MyConvTranspose2d(nn.Module):
    def __init__(self, conv, output_size):
        super().__init__()
        self.output_size = output_size
        self.conv = conv

    def forward(self, x):
        x = self.conv(x, output_size=self.output_size)
        return x


class Decoder_ReacherD4(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim

        self.first = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Unflatten(dim=-1, unflattened_size=(-1, 1, 1)),
        )
        # Defined custom ConvTranspose2d to set the output_size
        self.enc_net = nn.Sequential(
            nn.Conv2d(np.prod(input_shape), 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            MyConvTranspose2d(
                nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1),
                output_size=(-1, 512, 8, 8),
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            MyConvTranspose2d(
                nn.ConvTranspose2d(512, 8, 3, stride=2, padding=1),
                output_size=(-1, 4, 32, 32),
            ),
        )
        self.ext_net = nn.Sequential(
            MyConvTranspose2d(
                nn.ConvTranspose2d(8, 32, 3, stride=2, padding=1),
                output_size=(-1, 6, 64, 64),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=1, dilation=32, padding=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=1, dilation=16, padding=16),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=1, dilation=4, padding=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            MyConvTranspose2d(
                nn.ConvTranspose2d(32, output_dim, 3, stride=2, padding=1),
                output_size=(-1, 6, 128, 128),
            ),
        )

    def forward(self, x):
        x = self.first(x)
        x = self.enc_net(x)
        x = self.ext_net(x)
        return x


class Decoder_CubesC4(torch.nn.Module):
    def __init__(self, input_shape, num_objects, output_shape, act_fn="relu"):
        super().__init__()
        self.input_shape = input_shape
        self.num_objects = num_objects
        self.output_shape = output_shape

        self.first = nn.Flatten(start_dim=1, end_dim=-1)

        self.fc1 = torch.nn.Linear(num_objects * np.prod(input_shape), 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, num_objects * output_shape[1] * output_shape[2])
        self.ln = torch.nn.LayerNorm(512)

        self.middle = nn.Unflatten(-1, (num_objects, output_shape[1], output_shape[2]))

        self.deconv1 = torch.nn.ConvTranspose2d(
            num_objects, 32, kernel_size=3, padding=1
        )
        self.deconv2 = torch.nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1)
        self.deconv3 = torch.nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1)
        self.deconv4 = torch.nn.ConvTranspose2d(
            32, output_shape[0], kernel_size=3, padding=1
        )

        self.ln1 = torch.nn.BatchNorm2d(32)
        self.ln2 = torch.nn.BatchNorm2d(32)
        self.ln3 = torch.nn.BatchNorm2d(32)

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)
        self.act3 = utils.get_act_fn(act_fn)
        self.act4 = utils.get_act_fn(act_fn)
        self.act5 = utils.get_act_fn(act_fn)

    def forward(self, x):
        # Encoder
        x = self.first(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.ln(self.fc2(x)))
        x = self.fc3(x)

        # Embedding
        x = self.middle(x)
        x = self.act3(self.ln1(self.deconv1(x)))
        x = self.act4(self.ln1(self.deconv2(x)))
        x = self.act5(self.ln1(self.deconv3(x)))
        x = self.deconv4(x)

        return x
