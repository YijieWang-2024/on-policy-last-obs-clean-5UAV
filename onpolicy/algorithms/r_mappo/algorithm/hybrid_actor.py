"""Evaluation-only actor assembled from two trained spatial-flight actors.

The flight encoder and flight action head come from ``flight_actor``.  The
resource base, association/resource heads, and association threshold come
from ``resource_actor``.  This module deliberately does not change the
normal R_Actor training path.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from onpolicy.algorithms.utils.util import check


class HybridRActor(nn.Module):
    """Combine a flight branch and a resource/association branch.

    The current project uses a feed-forward ``SpatialFlightEncoder`` with four
    mixed-action heads.  The class refuses unsupported recurrent or attention
    actors instead of silently combining incompatible interfaces.
    """

    expects_resource_obs = True

    def __init__(
        self,
        flight_actor: nn.Module,
        resource_actor: nn.Module,
        runtime_association_threshold: float | None = None,
    ) -> None:
        super().__init__()
        if getattr(flight_actor, "flight_base", None) is None:
            raise ValueError("flight_actor must use spatial_flight_actor")
        if getattr(resource_actor, "flight_base", None) is None:
            raise ValueError("resource_actor must use spatial_flight_actor")
        if getattr(flight_actor, "_use_recurrent_policy", False) or getattr(
            flight_actor, "_use_naive_recurrent_policy", False
        ):
            raise ValueError("HybridRActor currently supports feed-forward actors only")
        if getattr(resource_actor, "_use_recurrent_policy", False) or getattr(
            resource_actor, "_use_naive_recurrent_policy", False
        ):
            raise ValueError("HybridRActor currently supports feed-forward actors only")

        flight_act = flight_actor.act
        resource_act = resource_actor.act
        if not getattr(flight_act, "mixed_action", False) or not getattr(
            resource_act, "mixed_action", False
        ):
            raise ValueError("HybridRActor requires mixed continuous action heads")
        if flight_act.action_dims != resource_act.action_dims:
            raise ValueError(
                "Flight/resource action-head dimensions differ: "
                f"{flight_act.action_dims} vs {resource_act.action_dims}"
            )
        if len(flight_act.action_outs) < 4:
            raise ValueError("HybridRActor expects flight, association, bandwidth, and CPU heads")

        self.flight_base = flight_actor.flight_base
        self.resource_base = resource_actor.base
        self._actor_message_dim = int(getattr(resource_actor, "_actor_message_dim", 0))
        self._actor_message_start = int(getattr(resource_actor, "_actor_message_start", 2))
        self._spatial_flight_actor = True
        self.hidden_size = int(resource_actor.hidden_size)
        self.tpdv = dict(resource_actor.tpdv)

        # The resource ACTLayer owns the runtime threshold and the resource
        # heads. Replace only its flight head with the flight checkpoint head.
        self.act = deepcopy(resource_act)
        self.act.action_outs[0] = deepcopy(flight_act.action_outs[0])
        if runtime_association_threshold is not None:
            threshold = float(runtime_association_threshold)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("runtime association threshold must be in [0, 1]")
            self.act.association_threshold = threshold

    def _resource_actor_obs(self, obs):
        if not self._actor_message_dim:
            return obs
        message_end = self._actor_message_start + self._actor_message_dim
        return torch.cat((obs[:, :self._actor_message_start], obs[:, message_end:]), dim=-1)

    def _features(self, flight_obs, resource_obs, available_actions):
        flight_obs = check(flight_obs).to(**self.tpdv)
        resource_obs = check(resource_obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        resource_features = self.resource_base(self._resource_actor_obs(resource_obs))
        flight_features = self.flight_base(flight_obs, available_actions)
        return resource_features, flight_features, available_actions

    def forward(
        self,
        obs,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False,
        resource_obs=None,
        **_,
    ):
        if resource_obs is None:
            raise ValueError("HybridRActor.forward requires resource_obs")
        rnn_states = check(rnn_states).to(**self.tpdv)
        check(masks).to(**self.tpdv)
        resource_features, flight_features, available_actions = self._features(
            obs, resource_obs, available_actions
        )
        actions, action_log_probs = self.act(
            resource_features,
            available_actions,
            deterministic,
            flight_x=flight_features,
        )
        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self,
        obs,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        resource_obs=None,
        **_,
    ):
        if resource_obs is None:
            raise ValueError("HybridRActor.evaluate_actions requires resource_obs")
        resource_features, flight_features, available_actions = self._features(
            obs, resource_obs, available_actions
        )
        return self.act.evaluate_actions(
            resource_features,
            action,
            available_actions,
            active_masks=active_masks,
            flight_x=flight_features,
        )
