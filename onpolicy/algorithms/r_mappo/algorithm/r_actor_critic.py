import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.utils.util import get_shape_from_obs_space


class SpatialFlightEncoder(nn.Module):
    """Task-free, permutation-invariant encoder for the local flight policy."""

    def __init__(self, args, obs_shape):
        super().__init__()
        if not getattr(args, "dynamic_md", False):
            raise ValueError("spatial_flight_actor currently requires dynamic_md")
        if getattr(args, "actor_neighbor_obs", False):
            raise ValueError("spatial_flight_actor intentionally excludes neighbor observations")

        self.max_users = args.max_GUs_in_range
        self.user_stride = 10
        self.prefix = int(getattr(args, "ob_state_with_timestep", False))
        if getattr(args, "ob_state_with_id", False):
            self.prefix += args.n_UAVs
        self.users_start = self.prefix + 3  # own x/y and local-user count
        expected_dim = self.users_start + self.max_users * self.user_stride
        if obs_shape[0] != expected_dim:
            raise ValueError(
                f"spatial_flight_actor expected obs dim {expected_dim}, got {obs_shape[0]}"
            )

        hidden = args.hidden_size
        self.user_encoder = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden + 3, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, obs, available_actions):
        own_position = obs[:, self.prefix:self.prefix + 2]
        users = obs[:, self.users_start:].reshape(
            obs.shape[0], self.max_users, self.user_stride
        )
        # Spatial/mobility/session fields only: x, y, speed, direction,
        # Gaussian-Markov base direction and remaining lifetime. Channel and
        # current i.i.d. task fields are deliberately excluded.
        user_features = users[:, :, :6]
        if available_actions is None:
            mask = torch.ones(
                user_features.shape[:2], dtype=obs.dtype, device=obs.device
            )
        else:
            mask = (available_actions > 0).to(obs.dtype)
        encoded = self.user_encoder(user_features) * mask.unsqueeze(-1)
        count = mask.sum(dim=1, keepdim=True)
        pooled = encoded.sum(dim=1) / count.clamp(min=1.0)
        occupancy = count / float(self.max_users)
        return self.readout(torch.cat((own_position, occupancy, pooled), dim=-1))


class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._layer_N = args.layer_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        obs_shape = get_shape_from_obs_space(obs_space)
        self._spatial_flight_actor = getattr(args, "spatial_flight_actor", False)
        if self._spatial_flight_actor:
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                raise ValueError("spatial_flight_actor currently supports feed-forward policies only")
            self.flight_base = SpatialFlightEncoder(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.base = MLPBase(args, obs_shape, layer_N=self._layer_N - 1)
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        else:
            self.base = MLPBase(args, obs_shape, layer_N=self._layer_N)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)
        self.to(device)
        self.algo = args.algorithm_name

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        actor_features = self.base(obs)
        flight_features = (
            self.flight_base(obs, available_actions)
            if self._spatial_flight_actor else None
        )

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic,
            flight_x=flight_features,
        )
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        actor_features = self.base(obs)
        flight_features = (
            self.flight_base(obs, available_actions)
            if self._spatial_flight_actor else None
        )
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                 action, available_actions,
                                                                 active_masks=
                                                                 active_masks if self._use_policy_active_masks
                                                                 else None,
                                                                 flight_x=flight_features)
        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._layer_N = args.layer_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.base = MLPBase(args, cent_obs_shape, layer_N=self._layer_N - 1)
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        else:
            self.base = MLPBase(args, cent_obs_shape, layer_N=self._layer_N)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)
        return values, rnn_states
