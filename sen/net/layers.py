import torch


class C4Conv(torch.nn.Module):
    """C_4 Convolution where CCW 90deg -> torch.roll(-1) along fiber dimension (opposite of e2cnn)"""

    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.rand(4, size_out, size_in)
        weights -= 0.5
        k = 1 / torch.sqrt(torch.tensor(size_in, dtype=torch.float))
        weights *= k
        self.weights = torch.nn.parameter.Parameter(weights)
        # bias = torch.rand(size_out)
        # bias -= 0.5
        # bias *= k
        bias = torch.zeros(size_out)
        self.bias = torch.nn.parameter.Parameter(bias)
        mat = torch.stack(
            [torch.roll(self.weights, i, dims=0) for i in range(4)], dim=0
        )
        self.register_buffer("mat", mat)

    def updateKernel(self):
        # V1,V2
        self.mat = torch.stack(
            [torch.roll(self.weights, i, dims=0) for i in range(4)], dim=0
        )

    def forward(self, x):
        # really should only call after update
        self.updateKernel()

        # x:  [B,O,4,size_in]
        # mat:[4,4,size_out,size_in]
        # y:  [B,O,4,size_out]
        w_times_x = torch.einsum("ghij,...hj->...gi", self.mat, x)

        return torch.add(w_times_x, self.bias)  # w times x + b


## We denote the representations of the group C_4 = {1,g,g^2,g^3} as follows:
# rhoreg - R^4 with permutation action  g e_i = e_{i+1 mod 4}
# triv - R with g e_1 = e_1
# sign - R with g e_1 = -e_1
# std - R^2 with g = {{0,-1}
#                     {1, 0}}

# {{x,-y,-x, y}
#   y, x,-y,-x}}
class C4_rhoreg_to_std_Conv(torch.nn.Module):
    """C_4 Convolution"""

    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.rand(2, size_out, size_in)
        weights -= 0.5
        k = 1 / torch.sqrt(torch.tensor(size_in, dtype=torch.float))
        weights *= k
        self.weights = torch.nn.parameter.Parameter(weights)
        mat = torch.stack(
            [
                torch.stack(
                    [
                        self.weights[0],
                        -self.weights[1],
                        -self.weights[0],
                        self.weights[1],
                    ]
                ),
                torch.stack(
                    [
                        self.weights[1],
                        self.weights[0],
                        -self.weights[1],
                        -self.weights[0],
                    ]
                ),
            ]
        )
        self.register_buffer("mat", mat)

    def updateKernel(self):
        self.mat = torch.stack(
            [
                torch.stack(
                    [
                        self.weights[0],
                        -self.weights[1],
                        -self.weights[0],
                        self.weights[1],
                    ]
                ),
                torch.stack(
                    [
                        self.weights[1],
                        self.weights[0],
                        -self.weights[1],
                        -self.weights[0],
                    ]
                ),
            ]
        )

    def forward(self, x):
        self.updateKernel()
        # x:  [B,O,4,size_in]
        # mat:[2,4,size_out,size_in]
        # y:  [B,O,2,size_out]
        w_times_x = torch.einsum("ghij,...hj->...gi", self.mat, x)

        return w_times_x
