import torch


class Model(torch.nn.Module):
    def __init__(
        self,
        embedding,
        encoder,
        transition,
        hinge=1.0,
        sigma=0.5,
        n_neg=1,
        pred_delta=True,
    ):
        super().__init__()

        self.embedding = embedding
        self.encoder = encoder
        self.transition = transition
        self.hinge = hinge
        self.sigma = sigma
        self.n_neg = int(n_neg)
        # If true, predict delta_z in transition model
        self.pred_delta = pred_delta

    def _energy(self, inp, target):
        """
        Energy distance based on normalized squared L2 norm
        inp: shape = [B, ...]
        target: shape = [N_neg, B, ...]
        returns: (inp-target) sum-reduced to shape [B]
        """
        # diff.shape == target.shape
        diff = (inp - target).pow(2)

        # If there are multiple negative samples, take the mean over neg samples
        if inp.shape != target.shape:
            diff = diff.mean(0)

        # Take average over object dimension
        if hasattr(self.transition, "num_objects"):
            diff = diff.mean(1)

        # Sum over all remaining dims except for batch_dim
        sum_dims = tuple(range(1, diff.dim()))
        diff = diff.sum(sum_dims)
        assert diff.shape == (inp.shape[0],)

        norm = 0.5 / (self.sigma**2)
        return norm * diff

    def contrastive_loss(self, obs, action, next_obs):
        state, trans_out, next_state = self(obs, action, next_obs)
        # Positive loss
        if self.pred_delta:
            pos_loss = self._energy(state + trans_out, next_state)
        else:
            pos_loss = self._energy(trans_out, next_state)

        # Negative loss
        # Sample negative state across episodes at random
        batch_size = state.shape[0]
        perm = torch.stack([torch.randperm(batch_size) for _ in range(self.n_neg)])
        neg_state = state[perm]
        neg_loss = torch.max(
            torch.zeros_like(pos_loss),
            self.hinge - self._energy(state, neg_state),
        )

        loss = pos_loss.mean() + neg_loss.mean()

        with torch.no_grad():
            if hasattr(self.transition, "num_objects"):
                z_norm = torch.linalg.vector_norm(state.mean(1).mean(0))
            else:
                z_norm = torch.linalg.vector_norm(state.mean(0))

        return loss, z_norm

    def forward(self, obs, action, next_obs):
        enc = self.embedding(obs)
        next_enc = self.embedding(next_obs)

        state = self.encoder(enc)
        next_state = self.encoder(next_enc)

        trans_out = self.transition(state, action)

        return state, trans_out, next_state
