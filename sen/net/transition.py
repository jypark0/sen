import numpy as np
import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as e2nn

from sen import utils
from sen.net.layers import C4Conv


class Transition(nn.Module):
    """Reacher Transition function (no GNN)."""

    def __init__(self, obs_dim, action_dim, hidden_dim=[128, 128]):
        # Transition(obs=8,act=2,hid=[128,128]) = 18952

        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        assert len(hidden_dim) == 2, "len(hidden_dim) must equal 2"
        self.hidden_dim = hidden_dim

        self.first = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
        )

        self.net = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.LayerNorm(hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], self.obs_dim),
        )

    def forward(self, z, actions):
        # z: (B, dim)
        # actions: (B, 2)
        z = self.first(z)
        a = self.first(actions)

        z_a = torch.cat([z, a], dim=1)

        delta_z = self.net(z_a)

        return delta_z


class Transition_D4(nn.Module):
    """Reacher D4 transition function (no GNN)."""

    def __init__(self, obs_dim, action_dim, group_order, hidden_dim=[64, 64]):

        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        assert len(hidden_dim) == 2, "len(hidden_dim) must equal 2"
        self.hidden_dim = hidden_dim

        self.r2_act = gspaces.FlipRot2dOnR2(N=group_order)

        self.feat_type_obs = e2nn.FieldType(
            self.r2_act, obs_dim * [self.r2_act.regular_repr]
        )
        self.feat_type_act = e2nn.FieldType(
            self.r2_act, action_dim * [self.r2_act.quotient_repr((None, group_order))]
        )
        self.feat_type_in = e2nn.FieldType(
            self.r2_act,
            obs_dim * [self.r2_act.regular_repr]
            + action_dim * [self.r2_act.quotient_repr((None, group_order))],
        )
        self.feat_type_hid = [
            e2nn.FieldType(self.r2_act, h * [self.r2_act.regular_repr])
            for h in self.hidden_dim
        ]
        self.feat_type_out = e2nn.FieldType(
            self.r2_act, obs_dim * [self.r2_act.regular_repr]
        )

        # Shape: [B, ...] ->  [B, *, 1, 1]
        self.first = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Unflatten(dim=-1, unflattened_size=(-1, 1, 1)),
        )

        self.net = e2nn.SequentialModule(
            e2nn.R2Conv(self.feat_type_in, self.feat_type_hid[0], kernel_size=1),
            e2nn.ReLU(self.feat_type_hid[0]),
            e2nn.R2Conv(self.feat_type_hid[0], self.feat_type_hid[1], kernel_size=1),
            e2nn.ReLU(self.feat_type_hid[1]),
            e2nn.R2Conv(self.feat_type_hid[1], self.feat_type_out, kernel_size=1),
        )

        # Shape: [B, obs_dim*2*group_order, 1, 1] ->  [B, obs_dim, 2*group_order]
        self.last = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Unflatten(dim=-1, unflattened_size=(obs_dim, 2 * group_order)),
        )

    def forward(self, z, actions):
        # states: (B, dim)
        # actions: (B, 2)
        # Need to unsqueeze to 4-dim tensors
        z = self.first(z)
        z = e2nn.GeometricTensor(z, self.feat_type_obs)

        a = torch.stack([actions, -actions], dim=-1).view(-1, self.action_dim * 2, 1, 1)
        a = self.first(a)
        a = e2nn.GeometricTensor(a, self.feat_type_act)
        z_a = e2nn.tensor_directsum([z, a])

        # out can be delta_z or z' (depends on which model function is called)
        out = self.net(z_a)

        # Shape back into [B, obs_dim, group_order]
        out = self.last(out.tensor)

        return out


class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = torch.nn.Parameter(torch.FloatTensor(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_dim)
            + " -> "
            + str(self.out_dim)
            + ")"
        )


class TransitionGCN(nn.Module):
    """GCN-based transition function."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        num_objects,
        hidden_dim=512,
        action_type="discrete",
        ignore_action=False,
        copy_action=False,
        act_fn="relu",
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.action_type = action_type
        self.ignore_action = ignore_action
        self.copy_action = copy_action

        if self.ignore_action:
            self.action_dim = 0
        else:
            self.action_dim = action_dim

        node_input_dim = self.obs_dim + self.action_dim

        self.gc1 = GraphConvLayer(node_input_dim, hidden_dim)
        self.act = utils.get_act_fn(act_fn)
        self.gc2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.gc3 = GraphConvLayer(hidden_dim, self.obs_dim)

        self.batch_size = None
        self.adj = None

    def _node_model(self, node_attr, adj):
        node_attr = self.act(self.gc1(node_attr, adj))
        node_attr = self.act(self.ln(self.gc2(node_attr, adj)))
        node_attr = self.gc3(node_attr, adj)

        return node_attr

    def _get_adj_mat_fully_connected(self, batch_size, num_objects, device):
        B = batch_size
        O = num_objects

        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.adj is None or self.batch_size != B:
            self.batch_size = B

            # Create fully-connected normalized symmetric adjacency matrix for single sample
            self.adj = torch.ones(O, O, device=device) / O

            # Stack `batch_size` times in block-diagonal fashion
            self.adj = torch.block_diag(*([self.adj] * B))

        return self.adj

    def forward(self, states, action):
        device = states.device
        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.view(-1, self.obs_dim)

        adj = self._get_adj_mat_fully_connected(batch_size, num_nodes, device)

        if not self.ignore_action:

            if self.action_type == "discrete":
                if self.copy_action:
                    action_vec = utils.to_one_hot(action, self.action_dim).repeat(
                        1, self.num_objects
                    )
                    action_vec = action_vec.view(-1, self.action_dim)
                else:
                    action_vec = utils.to_one_hot(action, self.action_dim * num_nodes)
                    action_vec = action_vec.view(-1, self.action_dim)

            elif self.action_type == "continuous_invariant":
                action_vec = action.unsqueeze(1).repeat(1, self.num_objects, 1)
                action_vec = action_vec.view(-1, self.action_dim)
            else:
                raise NotImplementedError

            # Attach action to each state
            node_attr = torch.cat([node_attr, action_vec], dim=-1)

        node_attr = self._node_model(node_attr, adj)

        # [batch_size, num_nodes, hidden_dim]
        node_attr = node_attr.view(batch_size, num_nodes, -1)

        return node_attr


class TransitionGNN(nn.Module):
    """GNN-based transition function."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        num_objects,
        hidden_dim=512,
        action_type="discrete",
        ignore_action=False,
        copy_action=False,
        act_fn="relu",
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.action_type = action_type
        self.ignore_action = ignore_action
        self.copy_action = copy_action

        if self.ignore_action:
            self.action_dim = 0
        else:
            self.action_dim = action_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.obs_dim * 2, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
        )

        node_input_dim = hidden_dim + self.obs_dim + self.action_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, self.obs_dim),
        )

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(
        self, source, target, edge_attr, source_indices=None, target_indices=None
    ):
        del edge_attr  # Unused.
        out = torch.cat([source, target], dim=1)

        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, _ = edge_index
            agg = utils.unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0)
            )
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr

        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects, device):
        B = batch_size
        O = num_objects
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != B:
            self.batch_size = B

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(O, O, device=device) - torch.eye(O, device=device)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(B, 1)
            offset = torch.arange(0, B * O, O, device=device).repeat_interleave(
                O * (O - 1)
            )
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)

        return self.edge_list

    def forward(self, states, action):
        device = states.device
        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.view(-1, self.obs_dim)

        edge_attr = None
        edge_index = None

        if num_nodes > 1:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, device
            )

            row, col = edge_index
            edge_attr = self._edge_model(
                node_attr[row],
                node_attr[col],
                edge_attr,
                source_indices=row % self.num_objects,
                target_indices=col % self.num_objects,
            )

        if not self.ignore_action:

            if self.action_type == "discrete":
                if self.copy_action:
                    action_vec = utils.to_one_hot(action, self.action_dim).repeat(
                        1, self.num_objects
                    )
                    action_vec = action_vec.view(-1, self.action_dim)
                else:
                    action_vec = utils.to_one_hot(action, self.action_dim * num_nodes)
                    action_vec = action_vec.view(-1, self.action_dim)

            elif self.action_type == "continuous_invariant":
                action_vec = action.unsqueeze(1).repeat(1, self.num_objects, 1)
                action_vec = action_vec.view(-1, self.action_dim)
            else:
                raise NotImplementedError

            # Attach action to each state
            node_attr = torch.cat([node_attr, action_vec], dim=-1)

        node_attr = self._node_model(node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        node_attr = node_attr.view(batch_size, num_nodes, -1)

        return node_attr


class TransitionGNN_C4(nn.Module):
    """GNN-based transition function."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        num_objects,
        hidden_dim=256,
        action_type="discrete",
        ignore_action=False,
        copy_action=False,
        act_fn="relu",
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.action_type = action_type
        self.ignore_action = ignore_action
        self.copy_action = copy_action

        if self.ignore_action:
            self.action_dim = 0
        else:
            self.action_dim = action_dim

        self.edge_mlp = nn.Sequential(
            C4Conv(self.obs_dim * 2, hidden_dim),
            utils.get_act_fn(act_fn),
            C4Conv(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            C4Conv(hidden_dim, hidden_dim),
        )

        if self.action_type == "discrete":
            node_input_dim = hidden_dim + self.obs_dim + 1
        elif self.action_type == "continuous_invariant":
            node_input_dim = hidden_dim + self.obs_dim + self.action_dim

        self.node_mlp = nn.Sequential(
            C4Conv(node_input_dim, hidden_dim),
            utils.get_act_fn(act_fn),
            C4Conv(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            C4Conv(hidden_dim, self.obs_dim),
        )

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target, edge_attr):
        del edge_attr  # Unused.
        out = torch.cat([source, target], dim=2)

        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, _ = edge_index
            agg = utils.unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0)
            )
            out = torch.cat([node_attr, agg], dim=2)
        else:
            out = node_attr

        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects, device):
        B = batch_size
        O = num_objects
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != B:
            self.batch_size = B

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(O, O, device=device) - torch.eye(O, device=device)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(B, 1)
            offset = torch.arange(0, B * O, O, device=device).repeat_interleave(
                O * (O - 1)
            )
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)

        return self.edge_list

    def forward(self, states, action):
        # States shape: [B, O, 4, 2]
        assert len(states.shape) == 4
        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [B, O, 4, 2]
        # node_attr: Flatten states tensor to [B * O, 4, 2]
        node_attr = states.reshape(-1, 4, self.obs_dim).contiguous()

        edge_attr = None
        edge_index = None

        if num_nodes > 1:
            # edge_index: [2, B * O * (O-1)] edge list
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, states.device
            )
            row, col = edge_index
            # [B* O*(O-1), 4, hidden_dim]
            edge_attr = self._edge_model(
                node_attr[row],
                node_attr[col],
                edge_attr,
            )

        if not self.ignore_action:
            if self.action_type == "discrete":
                if self.copy_action:
                    action_vec = utils.to_one_hot(action, self.action_dim).repeat(
                        1, self.num_objects
                    )
                    action_vec = action_vec.view(-1, self.action_dim)
                else:
                    action_vec = utils.to_one_hot(action, self.action_dim * num_nodes)
                    action_vec = action_vec.view(-1, self.action_dim)

            elif self.action_type == "continuous_invariant":
                action_vec = action.unsqueeze(1).repeat(1, self.num_objects, 1)
                action_vec = action_vec.view(-1, self.action_dim)
            else:
                raise NotImplementedError

            # Attach action to each state
            node_attr = torch.cat([node_attr, action_vec.unsqueeze(2)], dim=-1)

        # [B*O,4,obs=2+act=1]
        node_attr = self._node_model(node_attr, edge_index, edge_attr)

        # [B, O, 4, 2]
        node_attr = node_attr.view(batch_size, num_nodes, 4, -1)

        return node_attr
