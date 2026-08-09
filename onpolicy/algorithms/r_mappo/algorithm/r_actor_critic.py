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
        self.message_mode = getattr(args, "actor_message_mode", "disabled")
        self.message_pool = getattr(args, "actor_message_pool", "mean")
        if self.message_pool not in {"mean", "receiver_gated_sum"}:
            raise ValueError(f"unknown actor_message_pool: {self.message_pool}")
        self.message_features = 10
        self.message_count = args.n_UAVs - 1
        self.message_dim = (
            self.message_features * self.message_count
            if self.message_mode != "disabled"
            else 0
        )
        self.layout_context_dim = (
            8 if getattr(args, "episode_layout_context", False) else 0
        )
        self.messages_start = self.prefix + 2
        self.context_start = self.messages_start + self.message_dim
        self.users_start = (
            self.context_start + self.layout_context_dim + 1
        )
        expected_dim = self.users_start + self.max_users * self.user_stride
        if obs_shape[0] != expected_dim:
            raise ValueError(
                f"spatial_flight_actor expected obs dim {expected_dim}, got {obs_shape[0]}"
            )

        hidden = args.hidden_size
        self.hidden_size = hidden
        self.user_encoder = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        if self.message_dim:
            self.message_encoder = nn.Sequential(
                nn.Linear(self.message_features - 1, hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
        self.readout = nn.Sequential(
            nn.Linear(
                hidden
                + 3
                + self.layout_context_dim
                + (hidden + 1 if self.message_dim else 0),
                hidden,
            ),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.message_gate = None

    def initialize_message_pool(self):
        """Create optional parameters after all shared actor modules.

        R_Actor calls this after constructing its resource base and action
        heads. Consequently, resetting the same torch seed gives mean and
        gated variants identical shared parameters; the extra gate cannot
        perturb the matched R0 control through initialization order.
        """
        if self.message_pool != "receiver_gated_sum" or not self.message_dim:
            return
        gate_input_dim = 2 * self.hidden_size + 5
        # The five policies are built sequentially from one global RNG stream.
        # Isolate the additional gate initialization so a matched gated run
        # retains identical critic and later-agent initialization.
        with torch.random.fork_rng(devices=[]):
            self.message_gate = nn.Sequential(
                nn.Linear(gate_input_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1),
            )
        # Begin close to the no-message residual while preserving gradients.
        nn.init.constant_(self.message_gate[-1].bias, -2.1972245773362196)

    def _pool_messages(self, messages, self_descriptor):
        message_mask = messages[:, :, -1].clamp(min=0.0, max=1.0)
        encoded_messages = self.message_encoder(messages[:, :, :-1])
        message_count = message_mask.sum(dim=1, keepdim=True)
        message_fraction = message_count / float(self.message_count)
        if self.message_pool == "mean":
            masked_messages = encoded_messages * message_mask.unsqueeze(-1)
            pooled_messages = masked_messages.sum(dim=1) / message_count.clamp(
                min=1.0
            )
            return pooled_messages, message_fraction

        if self.message_gate is None:
            raise RuntimeError("receiver-gated message pool was not initialized")
        receiver_context = self_descriptor.unsqueeze(1).expand(
            -1, self.message_count, -1
        )
        gate_inputs = torch.cat(
            (receiver_context, encoded_messages, messages[:, :, :2]), dim=-1
        )
        independent_gates = torch.sigmoid(
            self.message_gate(gate_inputs).squeeze(-1)
        ) * message_mask
        pooled_messages = (
            encoded_messages * independent_gates.unsqueeze(-1)
        ).sum(dim=1) / float(self.message_count)
        return pooled_messages, message_fraction

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
        self_descriptor = torch.cat((own_position, occupancy, pooled), dim=-1)
        features = [self_descriptor]
        if self.layout_context_dim:
            features.append(
                obs[
                    :, self.context_start:self.context_start + self.layout_context_dim
                ]
            )
        if self.message_dim:
            messages = obs[
                :, self.messages_start:self.messages_start + self.message_dim
            ].reshape(obs.shape[0], self.message_count, self.message_features)
            pooled_messages, message_fraction = self._pool_messages(
                messages, self_descriptor
            )
            features.extend((message_fraction, pooled_messages))
        return self.readout(torch.cat(features, dim=-1))


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
        self._actor_message_mode = getattr(args, "actor_message_mode", "disabled")
        self._actor_message_dim = (
            10 * (args.n_UAVs - 1)
            if self._actor_message_mode != "disabled"
            else 0
        )
        self._actor_message_start = (
            int(getattr(args, "ob_state_with_timestep", False))
            + (args.n_UAVs if getattr(args, "ob_state_with_id", False) else 0)
            + 2
        )
        if self._spatial_flight_actor:
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                raise ValueError("spatial_flight_actor currently supports feed-forward policies only")
            self.flight_base = SpatialFlightEncoder(args, obs_shape)

        base_obs_shape = (
            (obs_shape[0] - self._actor_message_dim,)
            if self._actor_message_dim
            else obs_shape
        )

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.base = MLPBase(args, base_obs_shape, layer_N=self._layer_N - 1)
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        else:
            self.base = MLPBase(args, base_obs_shape, layer_N=self._layer_N)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)
        if self._spatial_flight_actor:
            self.flight_base.initialize_message_pool()
        self.to(device)
        self.algo = args.algorithm_name

    def _resource_actor_obs(self, obs):
        """Keep communicated summaries exclusive to the spatial flight head."""
        if not self._actor_message_dim:
            return obs
        message_end = self._actor_message_start + self._actor_message_dim
        return torch.cat(
            (obs[:, :self._actor_message_start], obs[:, message_end:]), dim=-1
        )

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        actor_features = self.base(self._resource_actor_obs(obs))
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
        actor_features = self.base(self._resource_actor_obs(obs))
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
