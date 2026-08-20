from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from onpolicy.utils.md_roster import (
    order_md_candidates,
    resolve_distance_only_user_sort,
)


class MDGRUPredictor(nn.Module):
    """One receiver-local mobility model whose parameters are shared by MD IDs."""

    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRUCell(feature_dim + 1, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
        )
        # A fresh model is exactly last-observation reconstruction. Learned
        # residuals can improve that baseline without exposing random absolute
        # positions to the critic.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def update(self, feature, age, hidden):
        return self.gru(torch.cat((feature, age), dim=-1), hidden)

    def predict(self, hidden, age, baseline=None):
        output = self.head(torch.cat((hidden, age), dim=-1))
        return output if baseline is None else baseline + output


class _RolloutSessionBuffer:
    """Dense rollout-local MD observation sequences, indexed by session ID."""

    def __init__(self, environments, session_capacity, max_events, feature_dim):
        self.environments = int(environments)
        self.session_capacity = int(session_capacity)
        self.max_events = int(max_events)
        self.feature_dim = int(feature_dim)
        self.features = np.zeros(
            (self.environments, self.session_capacity, self.max_events, feature_dim),
            dtype=np.float32,
        )
        self.gaps = np.zeros(
            (self.environments, self.session_capacity, self.max_events),
            dtype=np.uint16,
        )
        self.lengths = np.zeros(
            (self.environments, self.session_capacity), dtype=np.uint16
        )

    def ensure_capacity(self, required):
        required = int(required)
        if required <= self.session_capacity:
            return
        new_capacity = max(required, 2 * self.session_capacity)
        features = np.zeros(
            (self.environments, new_capacity, self.max_events, self.feature_dim),
            dtype=np.float32,
        )
        gaps = np.zeros(
            (self.environments, new_capacity, self.max_events), dtype=np.uint16
        )
        lengths = np.zeros((self.environments, new_capacity), dtype=np.uint16)
        features[:, :self.session_capacity] = self.features
        gaps[:, :self.session_capacity] = self.gaps
        lengths[:, :self.session_capacity] = self.lengths
        self.features, self.gaps, self.lengths = features, gaps, lengths
        self.session_capacity = new_capacity

    def reset(self, environment_mask=None):
        if environment_mask is None:
            self.lengths.fill(0)
        else:
            self.lengths[np.asarray(environment_mask, dtype=bool)] = 0

    def append(self, environments, session_ids, features, gaps):
        environments = np.asarray(environments, dtype=np.int64)
        session_ids = np.asarray(session_ids, dtype=np.int64)
        if not len(session_ids):
            return
        self.ensure_capacity(int(np.max(session_ids)) + 1)
        positions = self.lengths[environments, session_ids].astype(np.int64)
        if np.any(positions >= self.max_events):
            raise RuntimeError("MD session observation count exceeded md_lifetime_max")
        self.features[environments, session_ids, positions] = np.asarray(
            features, dtype=np.float32
        )
        self.gaps[environments, session_ids, positions] = np.asarray(
            gaps, dtype=np.uint16
        )
        self.lengths[environments, session_ids] += 1

    def session_pairs(self):
        environments, session_ids = np.nonzero(self.lengths >= 2)
        return np.column_stack((environments, session_ids)).astype(np.int64)

    def target_count(self, pairs=None):
        if pairs is None:
            return int(np.sum(np.maximum(self.lengths.astype(np.int64) - 1, 0)))
        if not len(pairs):
            return 0
        lengths = self.lengths[pairs[:, 0], pairs[:, 1]].astype(np.int64)
        return int(np.sum(lengths - 1))

    def take_target_budget(self, pairs, target_budget):
        if not len(pairs):
            return pairs
        counts = self.lengths[pairs[:, 0], pairs[:, 1]].astype(np.int64) - 1
        last = int(np.searchsorted(np.cumsum(counts), int(target_budget), side="left"))
        return pairs[:min(last + 1, len(pairs))]

    def age_values(self, pairs=None):
        if pairs is None:
            lengths, gaps = self.lengths, self.gaps
        elif len(pairs):
            lengths = self.lengths[pairs[:, 0], pairs[:, 1]]
            gaps = self.gaps[pairs[:, 0], pairs[:, 1]]
        else:
            return np.zeros(0, dtype=np.int64)
        steps = np.arange(self.max_events)
        mask = (steps >= 1) & (steps < lengths[..., None])
        return gaps[mask].astype(np.int64)

    def age_counts(self, pairs=None):
        ages = self.age_values(pairs)
        counts = np.zeros(4, dtype=np.int64)
        if len(ages):
            bins = np.minimum(np.maximum(ages, 1), 4) - 1
            counts += np.bincount(bins, minlength=4)[:4]
        return counts


class _DenseMemoryBank:
    """Receiver-local online beliefs with O(1) session-ID lookup."""

    def __init__(
        self, environments, capacity, session_capacity, feature_dim, hidden_dim, device
    ):
        self.environments = int(environments)
        self.capacity = int(capacity)
        self.session_capacity = int(session_capacity)
        self.device = device
        self.ids = np.full((environments, capacity), -1, dtype=np.int64)
        self.slot_by_session = np.full(
            (environments, session_capacity), -1, dtype=np.int16
        )
        self.last_seen = np.full((environments, capacity), -1, dtype=np.int32)
        self.expire_at = np.full((environments, capacity), -1, dtype=np.int32)
        self.last_feature = np.zeros(
            (environments, capacity, feature_dim), dtype=np.float32
        )
        # Tasks are current-slot data only. task_valid is cleared in begin_step.
        self.tasks = np.zeros((environments, capacity, 3), dtype=np.float32)
        self.task_valid = np.zeros((environments, capacity), dtype=bool)
        self.hidden = torch.zeros(
            environments, capacity, hidden_dim, device=device, dtype=torch.float32
        )

    def ensure_session_capacity(self, required):
        required = int(required)
        if required <= self.session_capacity:
            return
        new_capacity = max(required, 2 * self.session_capacity)
        lookup = np.full((self.environments, new_capacity), -1, dtype=np.int16)
        lookup[:, :self.session_capacity] = self.slot_by_session
        self.slot_by_session = lookup
        self.session_capacity = new_capacity

    def reset(self, environment_mask=None):
        if environment_mask is None:
            environment_mask = np.ones(self.environments, dtype=bool)
        environment_mask = np.asarray(environment_mask, dtype=bool)
        self.ids[environment_mask] = -1
        self.slot_by_session[environment_mask] = -1
        self.last_seen[environment_mask] = -1
        self.expire_at[environment_mask] = -1
        self.last_feature[environment_mask] = 0
        self.tasks[environment_mask] = 0
        self.task_valid[environment_mask] = False
        self.hidden[torch.as_tensor(environment_mask, device=self.device)] = 0

    def find(self, environment, session_id):
        if 0 <= session_id < self.session_capacity:
            slot = int(self.slot_by_session[environment, session_id])
            if slot >= 0 and self.ids[environment, slot] == session_id:
                return slot
        matches = np.flatnonzero(self.ids[environment] == session_id)
        if not len(matches):
            return None
        slot = int(matches[0])
        if session_id >= 0:
            self.ensure_session_capacity(session_id + 1)
            self.slot_by_session[environment, session_id] = slot
        return slot

    def begin_step(self, clock, protected_environments, protected_session_ids):
        self.task_valid.fill(False)
        expired = (self.ids != -1) & (self.expire_at <= clock[:, None])
        if len(protected_session_ids):
            self.ensure_session_capacity(int(np.max(protected_session_ids)) + 1)
            protected_slots = self.slot_by_session[
                protected_environments, protected_session_ids
            ].astype(np.int64)
            valid = protected_slots >= 0
            expired[protected_environments[valid], protected_slots[valid]] = False
        environments, slots = np.nonzero(expired)
        if not len(slots):
            return
        old_ids = self.ids[environments, slots]
        valid_ids = old_ids >= 0
        self.slot_by_session[environments[valid_ids], old_ids[valid_ids]] = -1
        self.ids[environments, slots] = -1
        self.last_seen[environments, slots] = -1
        self.expire_at[environments, slots] = -1
        self.last_feature[environments, slots] = 0
        self.tasks[environments, slots] = 0
        self.hidden[
            torch.as_tensor(environments, device=self.device),
            torch.as_tensor(slots, device=self.device),
        ] = 0

    def lookup_or_allocate(self, environments, session_ids):
        if not len(session_ids):
            return np.zeros(0, dtype=np.int64)
        self.ensure_session_capacity(int(np.max(session_ids)) + 1)
        slots = self.slot_by_session[environments, session_ids].astype(np.int64)
        missing = slots < 0
        for environment in np.unique(environments[missing]):
            indices = np.flatnonzero(missing & (environments == environment))
            free = np.flatnonzero(self.ids[environment] == -1)
            if len(free) < len(indices):
                raise RuntimeError("MD memory capacity was exceeded")
            assigned = free[:len(indices)]
            ids = session_ids[indices]
            slots[indices] = assigned
            self.ids[environment, assigned] = ids
            self.slot_by_session[environment, ids] = assigned
            self.last_seen[environment, assigned] = -1
            self.expire_at[environment, assigned] = -1
            self.hidden[environment, torch.as_tensor(assigned, device=self.device)] = 0
        return slots


class MDStateReconstructor:
    """Causal, batched receiver-ID memory and optional MD-GRU reconstruction."""

    feature_dim = 7
    checkpoint_version = 2

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.mode = args.state_reconstruction
        self.n_uavs = int(args.n_UAVs)
        arrivals = (
            int(sum(args.md_arrivals_per_region))
            if getattr(args, "md_arrivals_per_region", None) is not None
            else int(args.md_arrivals_max)
        )
        self.capacity = max(int(args.n_GUs), arrivals * int(args.md_lifetime_max))
        self.session_capacity = max(
            int(args.n_GUs), int(args.episode_length) * arrivals, 1
        )
        self.packet_capacity = int(args.max_GUs_in_range)
        self.hidden_dim = int(args.md_gru_hidden_dim)
        self.train_samples = int(getattr(args, "md_gru_train_samples", 512))
        self.min_ready_samples = int(
            getattr(args, "md_gru_min_ready_samples", 512)
        )
        if self.mode == "md_gru":
            if self.hidden_dim <= 0 or args.md_gru_max_samples <= 0:
                raise ValueError("md_gru_hidden_dim and md_gru_max_samples must be positive")
            if args.md_gru_lr <= 0 or args.md_gru_epochs <= 0 or args.md_gru_batch_size <= 0:
                raise ValueError(
                    "md_gru_lr, md_gru_epochs, and md_gru_batch_size must be positive"
                )
            if self.train_samples <= 0:
                raise ValueError("md_gru_train_samples must be positive")
            if self.min_ready_samples <= 0:
                raise ValueError("md_gru_min_ready_samples must be positive")
        self.distance_only_user_sort = resolve_distance_only_user_sort(args)
        self.metadata = bool(
            getattr(args, "critic_md_metadata", False) or self.mode != "zero"
        )
        predictor_template = (
            MDGRUPredictor(self.feature_dim, self.hidden_dim).to(device)
            if self.mode == "md_gru" else None
        )
        self.predictors = (
            nn.ModuleList([deepcopy(predictor_template) for _ in range(self.n_uavs)])
            if predictor_template is not None else None
        )
        self.optimizers = (
            [
                torch.optim.Adam(predictor.parameters(), lr=args.md_gru_lr)
                for predictor in self.predictors
            ]
            if self.predictors is not None else None
        )
        self.training_rng = np.random.default_rng(args.seed + 15485863)
        self.predictor_ready = (
            [False for _ in range(self.n_uavs)]
            if self.predictors is not None else [True for _ in range(self.n_uavs)]
        )
        self.residual_prediction = [True for _ in range(self.n_uavs)]
        self.banks = None
        self.session_buffers = None
        self.environments = None
        self.clock = None
        self.last_data = None
        self.query_age_counts = np.zeros((self.n_uavs, 4), dtype=np.int64)
        self._collect_metrics = False

    def checkpoint_state(self):
        if self.predictors is None:
            raise RuntimeError("only md_gru reconstruction has predictor state")
        return {
            "version": self.checkpoint_version,
            "prediction_contract": "last_obs_residual_v1",
            "models": self.predictors.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "predictor_ready": list(self.predictor_ready),
        }

    def load_checkpoint_state(self, checkpoint):
        """Load current residual checkpoints and legacy absolute-head checkpoints."""
        if self.predictors is None:
            raise RuntimeError("only md_gru reconstruction has predictor state")
        if isinstance(checkpoint, dict) and "models" in checkpoint:
            self.predictors.load_state_dict(checkpoint["models"])
            for optimizer, optimizer_state in zip(
                self.optimizers, checkpoint.get("optimizers", ())
            ):
                optimizer.load_state_dict(optimizer_state)
            ready = checkpoint.get("predictor_ready", [True] * self.n_uavs)
            self.predictor_ready = [bool(value) for value in ready]
            residual = checkpoint.get("prediction_contract") == "last_obs_residual_v1"
            self.residual_prediction = [residual for _ in range(self.n_uavs)]
        else:
            legacy_model = checkpoint.get("model", checkpoint)
            for predictor in self.predictors:
                predictor.load_state_dict(legacy_model)
            self.predictor_ready = [
                bool(checkpoint.get("predictor_ready", True))
                if isinstance(checkpoint, dict) else True
                for _ in range(self.n_uavs)
            ]
            self.residual_prediction = [False for _ in range(self.n_uavs)]
        self.predictors.eval()

    def _ensure_banks(self, environments):
        if self.banks is not None and self.environments == environments:
            return
        self.environments = int(environments)
        self.clock = np.zeros(self.environments, dtype=np.int32)
        self.banks = [
            _DenseMemoryBank(
                self.environments,
                self.capacity,
                self.session_capacity,
                self.feature_dim,
                self.hidden_dim,
                self.device,
            )
            for _ in range(self.n_uavs)
        ]
        self.session_buffers = (
            [
                _RolloutSessionBuffer(
                    self.environments,
                    self.session_capacity,
                    max(int(self.args.md_lifetime_max), 1),
                    self.feature_dim,
                )
                for _ in range(self.n_uavs)
            ]
            if self.predictors is not None else None
        )

    def _ensure_session_capacity(self, required):
        if required <= self.session_capacity:
            return
        self.session_capacity = max(int(required), 2 * self.session_capacity)
        for bank in self.banks:
            bank.ensure_session_capacity(self.session_capacity)
        if self.session_buffers is not None:
            for buffer in self.session_buffers:
                buffer.ensure_capacity(self.session_capacity)

    def reset(self, environment_mask=None):
        if self.banks is None:
            return
        if environment_mask is None:
            environment_mask = np.ones(self.environments, dtype=bool)
        environment_mask = np.asarray(environment_mask, dtype=bool)
        self.clock[environment_mask] = 0
        for bank in self.banks:
            bank.reset(environment_mask)

    def _speed_bound(self):
        candidates = [
            1.3 * float(self.args.mean_velocity),
            float(getattr(self.args, "md_velocity_init_max_factor", 1.3))
            * float(self.args.mean_velocity),
        ]
        update_clip_max = getattr(self.args, "md_velocity_update_clip_max", None)
        if update_clip_max is not None:
            candidates.append(float(update_clip_max))
        return max(*candidates, 1e-6)

    def _encode_features(self, records):
        records = np.asarray(records)
        x_span = max(self.args.x_max_gu - self.args.x_min_gu, 1e-6)
        y_span = max(self.args.y_max_gu - self.args.y_min_gu, 1e-6)
        speed_bound = self._speed_bound()
        features = np.empty((len(records), self.feature_dim), dtype=np.float32)
        features[:, 0] = 2 * (records[:, 0] - self.args.x_min_gu) / x_span - 1
        features[:, 1] = 2 * (records[:, 1] - self.args.y_min_gu) / y_span - 1
        features[:, 2] = 2 * records[:, 2] / speed_bound - 1
        features[:, 3] = np.sin(records[:, 3])
        features[:, 4] = np.cos(records[:, 3])
        features[:, 5] = np.sin(records[:, 4])
        features[:, 6] = np.cos(records[:, 4])
        return features

    def _encode_feature(self, record):
        """Compatibility wrapper for single-record diagnostics/tests."""
        return self._encode_features(np.asarray(record)[None])[0]

    def _decode_features(self, features):
        features = np.asarray(features, dtype=np.float32)
        x_span = max(self.args.x_max_gu - self.args.x_min_gu, 1e-6)
        y_span = max(self.args.y_max_gu - self.args.y_min_gu, 1e-6)
        speed_bound = self._speed_bound()
        decoded = np.empty(features.shape[:-1] + (5,), dtype=np.float32)
        decoded[..., 0] = (
            self.args.x_min_gu + 0.5 * (np.clip(features[..., 0], -1, 1) + 1) * x_span
        )
        decoded[..., 1] = (
            self.args.y_min_gu + 0.5 * (np.clip(features[..., 1], -1, 1) + 1) * y_span
        )
        decoded[..., 2] = (
            0.5 * (np.clip(features[..., 2], -1, 1) + 1) * speed_bound
        )
        decoded[..., 3] = np.mod(
            np.arctan2(features[..., 3], features[..., 4]), 2 * np.pi
        )
        decoded[..., 4] = np.mod(
            np.arctan2(features[..., 5], features[..., 6]), 2 * np.pi
        )
        return decoded

    def _decode_feature(self, feature):
        """Compatibility wrapper for single-feature diagnostics/tests."""
        return self._decode_features(np.asarray(feature)[None])[0]

    def _collect_observations(self, data):
        record_valid = np.asarray(data["record_valid"], dtype=bool)
        source_available = (
            np.asarray(data["geometric_mask"], dtype=bool)
            & np.asarray(data["reception_mask"], dtype=bool)
        )
        diagonal = np.arange(self.n_uavs)
        source_available[:, diagonal, diagonal] = True
        valid = source_available[..., None] & record_valid[:, None, :, :]
        environments, receivers, senders, record_slots = np.nonzero(valid)
        if not len(record_slots):
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, empty, np.zeros((0, 10), dtype=np.float32)
        session_ids = data["session_ids"][environments, senders, record_slots].astype(
            np.int64
        )
        valid_ids = session_ids >= 0
        environments = environments[valid_ids]
        receivers = receivers[valid_ids]
        senders = senders[valid_ids]
        record_slots = record_slots[valid_ids]
        session_ids = session_ids[valid_ids]
        if not len(session_ids):
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, empty, np.zeros((0, 10), dtype=np.float32)
        self._ensure_session_capacity(int(np.max(session_ids)) + 1)
        span = max(int(np.max(session_ids)) + 1, 1)
        keys = (environments * self.n_uavs + receivers) * span + session_ids
        _, first = np.unique(keys, return_index=True)
        return (
            environments[first].astype(np.int64),
            receivers[first].astype(np.int64),
            session_ids[first].astype(np.int64),
            np.asarray(
                data["record_features"][
                    environments[first], senders[first], record_slots[first]
                ],
                dtype=np.float32,
            ),
        )

    @torch.no_grad()
    def _ingest_receiver(
        self,
        receiver,
        observation_environments,
        observation_receivers,
        observation_session_ids,
        observation_records,
        record_environment_mask,
    ):
        selected = observation_receivers == receiver
        environments = observation_environments[selected]
        session_ids = observation_session_ids[selected]
        records = observation_records[selected]
        bank = self.banks[receiver]
        bank.begin_step(self.clock, environments, session_ids)
        observed_slots = np.zeros((self.environments, self.capacity), dtype=bool)
        if not len(session_ids):
            return observed_slots

        slots = bank.lookup_or_allocate(environments, session_ids)
        observed_slots[environments, slots] = True
        features = self._encode_features(records)
        old_last_seen = bank.last_seen[environments, slots]
        gaps = np.where(
            old_last_seen < 0, 0, self.clock[environments] - old_last_seen
        ).astype(np.int64)

        if self.session_buffers is not None:
            record = record_environment_mask[environments]
            self.session_buffers[receiver].append(
                environments[record], session_ids[record], features[record], gaps[record]
            )

        predictor = self.predictors[receiver] if self.predictors is not None else None
        if predictor is not None:
            environment_index = torch.as_tensor(
                environments, device=self.device, dtype=torch.long
            )
            slot_index = torch.as_tensor(slots, device=self.device, dtype=torch.long)
            feature_tensor = torch.as_tensor(
                features, device=self.device, dtype=torch.float32
            )
            age_tensor = torch.as_tensor(
                gaps[:, None] / max(self.args.md_lifetime_max, 1),
                device=self.device,
                dtype=torch.float32,
            )
            bank.hidden[environment_index, slot_index] = predictor.update(
                feature_tensor,
                age_tensor,
                bank.hidden[environment_index, slot_index],
            )

        bank.last_feature[environments, slots] = features
        bank.last_seen[environments, slots] = self.clock[environments]
        lifetime = np.maximum(np.rint(records[:, 5]).astype(np.int32), 1)
        bank.expire_at[environments, slots] = self.clock[environments] + lifetime
        bank.tasks[environments, slots] = records[:, 7:10]
        bank.task_valid[environments, slots] = True
        return observed_slots

    @torch.no_grad()
    def _current_features(self, data, observed_slots):
        geometric = np.asarray(data["geometric_mask"], dtype=bool)
        received = np.asarray(data["reception_mask"], dtype=bool)
        off_diagonal = ~np.eye(self.n_uavs, dtype=bool)
        missing = geometric & ~received & off_diagonal[None]
        needs_reconstruction = np.any(missing, axis=2)
        current = []
        for receiver, bank in enumerate(self.banks):
            features = bank.last_feature.copy()
            active = (bank.ids != -1) & (bank.expire_at > self.clock[:, None])
            query = (
                active
                & ~observed_slots[receiver]
                & needs_reconstruction[:, receiver, None]
            )
            environments, slots = np.nonzero(query)
            if len(slots):
                ages = self.clock[environments] - bank.last_seen[environments, slots]
                if self._collect_metrics:
                    bins = np.minimum(np.maximum(ages, 1), 4) - 1
                    self.query_age_counts[receiver] += np.bincount(
                        bins, minlength=4
                    )[:4]
                if self.predictors is not None and self.predictor_ready[receiver]:
                    environment_index = torch.as_tensor(
                        environments, device=self.device, dtype=torch.long
                    )
                    slot_index = torch.as_tensor(
                        slots, device=self.device, dtype=torch.long
                    )
                    age_tensor = torch.as_tensor(
                        ages[:, None] / max(self.args.md_lifetime_max, 1),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    baseline = torch.as_tensor(
                        bank.last_feature[environments, slots],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    if not self.residual_prediction[receiver]:
                        baseline = None
                    prediction = self.predictors[receiver].predict(
                        bank.hidden[environment_index, slot_index], age_tensor, baseline
                    )
                    features[environments, slots] = prediction.cpu().numpy()
            current.append(features)
        return np.stack(current, axis=0)

    def _channel_gain_from_horizontal(self, horizontal):
        vertical = self.args.H_UAV - self.args.H_GU
        distance = np.maximum(np.hypot(horizontal, vertical), 1e-10)
        theta = 180 / np.pi * np.arcsin(np.clip(vertical / distance, -1, 1))
        p_los = 1 / (1 + 9.61 * np.exp(-0.16 * (theta - 9.61)))
        log_term = 20 * np.log10(4 * np.pi * 2e9 / 3e8)
        path_loss = (
            p_los * (log_term + 1 + 20 * np.log10(distance))
            + (1 - p_los) * (log_term + 20 + 20 * np.log10(distance))
        )
        return np.power(10.0, -path_loss / 10).astype(np.float32)

    def _direct_blocks(self, data):
        prefix = np.asarray(data["critic_prefix"], dtype=np.float32)
        records = np.asarray(data["record_features"], dtype=np.float32)
        pieces = [prefix, records.reshape(self.environments, self.n_uavs, -1)]
        if self.metadata:
            valid = np.asarray(data["record_valid"], dtype=np.float32)
            metadata = np.stack((valid, valid, np.zeros_like(valid)), axis=-1)
            pieces.append(metadata.reshape(self.environments, self.n_uavs, -1))
        return np.concatenate(pieces, axis=-1)

    def _surrogate_prefixes(self, data, visible_count):
        prefix = np.asarray(data["critic_prefix"], dtype=np.float32)
        prefix_dim = prefix.shape[-1]
        receiver_prefix = np.transpose(prefix, (1, 0, 2))
        surrogate = np.broadcast_to(
            receiver_prefix[:, :, None, :],
            (self.n_uavs, self.environments, self.n_uavs, prefix_dim),
        ).copy()
        cursor = int(self.args.ob_state_with_timestep)
        if self.args.ob_state_with_id:
            surrogate[..., cursor:cursor + self.n_uavs] = np.eye(
                self.n_uavs, dtype=np.float32
            )[None, None]
            cursor += self.n_uavs
        surrogate[..., cursor:cursor + 2] = np.asarray(
            data["uav_positions"], dtype=np.float32
        )[None]
        surrogate[..., -1] = visible_count
        return surrogate

    def _surrogate_blocks(self, data, current_features):
        ids = np.stack([bank.ids for bank in self.banks], axis=0)
        last_seen = np.stack([bank.last_seen for bank in self.banks], axis=0)
        expire_at = np.stack([bank.expire_at for bank in self.banks], axis=0)
        tasks = np.stack([bank.tasks for bank in self.banks], axis=0)
        task_valid = np.stack([bank.task_valid for bank in self.banks], axis=0)
        active = (ids != -1) & (expire_at > self.clock[None, :, None])
        decoded = self._decode_features(current_features)
        sender_positions = np.asarray(data["uav_positions"], dtype=np.float32)
        delta = decoded[:, :, None, :, :2] - sender_positions[None, :, :, None, :]
        distances = np.linalg.norm(delta, axis=-1)
        eligible = active[:, :, None, :] & (distances <= self.args.Cover_R)
        ids_by_sender = np.broadcast_to(ids[:, :, None, :], eligible.shape)
        task_valid_by_sender = np.broadcast_to(
            task_valid[:, :, None, :], eligible.shape
        )
        can_finish = tasks[..., 1] / self.args.F_n <= tasks[..., 2]
        can_finish_by_sender = np.broadcast_to(
            can_finish[:, :, None, :], eligible.shape
        )
        order = order_md_candidates(
            np.where(eligible, distances, np.inf),
            np.where(eligible, ids_by_sender, np.iinfo(np.int64).max),
            can_finish_by_sender,
            task_valid_by_sender & eligible,
            distance_only=self.distance_only_user_sort,
        )
        selected_count = min(self.packet_capacity, self.capacity)
        selected = order[..., :selected_count]
        valid = np.take_along_axis(eligible, selected, axis=-1)
        selected_distances = np.take_along_axis(distances, selected, axis=-1)
        decoded_by_sender = np.broadcast_to(
            decoded[:, :, None, :, :],
            (self.n_uavs, self.environments, self.n_uavs, self.capacity, 5),
        )
        selected_decoded = np.take_along_axis(
            decoded_by_sender, selected[..., None], axis=3
        )
        remaining = np.maximum(
            expire_at - self.clock[None, :, None], 0
        ).astype(np.float32)
        age = np.maximum(
            self.clock[None, :, None] - last_seen, 0
        ).astype(np.float32)
        remaining_by_sender = np.broadcast_to(
            remaining[:, :, None, :], eligible.shape
        )
        age_by_sender = np.broadcast_to(age[:, :, None, :], eligible.shape)
        selected_remaining = np.take_along_axis(
            remaining_by_sender, selected, axis=-1
        )
        selected_age = np.take_along_axis(age_by_sender, selected, axis=-1)
        tasks_by_sender = np.broadcast_to(
            tasks[:, :, None, :, :],
            (self.n_uavs, self.environments, self.n_uavs, self.capacity, 3),
        )
        selected_tasks = np.take_along_axis(
            tasks_by_sender, selected[..., None], axis=3
        )
        selected_task_valid = np.take_along_axis(
            task_valid_by_sender, selected, axis=-1
        ) & valid

        records = np.zeros(
            (
                self.n_uavs,
                self.environments,
                self.n_uavs,
                self.packet_capacity,
                10,
            ),
            dtype=np.float32,
        )
        metadata = np.zeros(records.shape[:-1] + (3,), dtype=np.float32)
        records[..., :selected_count, :5] = selected_decoded * valid[..., None]
        records[..., :selected_count, 5] = selected_remaining * valid
        records[..., :selected_count, 6] = (
            self._channel_gain_from_horizontal(selected_distances) * valid
        )
        placeholder = np.asarray(
            [
                0.5 * (self.args.D_min + self.args.D_max),
                0.5 * (self.args.C_min + self.args.C_max),
                0.5 * (self.args.delay_min + self.args.delay_max),
            ],
            dtype=np.float32,
        )
        selected_task_values = np.where(
            selected_task_valid[..., None], selected_tasks, placeholder
        )
        records[..., :selected_count, 7:10] = selected_task_values * valid[..., None]
        metadata[..., :selected_count, 0] = valid
        metadata[..., :selected_count, 1] = selected_task_valid
        metadata[..., :selected_count, 2] = (
            selected_age / max(self.args.md_lifetime_max, 1) * valid
        )
        visible_count = np.sum(eligible, axis=-1, dtype=np.int32)
        pieces = [
            self._surrogate_prefixes(data, visible_count),
            records.reshape(self.n_uavs, self.environments, self.n_uavs, -1),
        ]
        if self.metadata:
            pieces.append(
                metadata.reshape(self.n_uavs, self.environments, self.n_uavs, -1)
            )
        return np.concatenate(pieces, axis=-1)

    def reconstruct(self, data, reset_environments=None, collect_samples=True):
        environments = int(data["record_features"].shape[0])
        self._ensure_banks(environments)
        if reset_environments is None:
            reset_environments = np.zeros(environments, dtype=bool)
        else:
            reset_environments = np.asarray(reset_environments, dtype=bool)
            self.reset(reset_environments)
        self.last_data = data
        self._collect_metrics = bool(collect_samples)
        # Auto-reset observations returned on a terminal training step belong to
        # the next rollout. They initialize online belief, but are appended to
        # the new session dataset only after the current predictor update.
        record_environment_mask = np.ones(environments, dtype=bool)
        if collect_samples:
            record_environment_mask[reset_environments] = False

        observation_data = self._collect_observations(data)
        observed_slots = [
            self._ingest_receiver(
                receiver, *observation_data, record_environment_mask
            )
            for receiver in range(self.n_uavs)
        ]
        current_features = self._current_features(data, observed_slots)
        direct = self._direct_blocks(data)
        surrogate = np.transpose(
            self._surrogate_blocks(data, current_features), (1, 0, 2, 3)
        )

        geometric = np.asarray(data["geometric_mask"], dtype=bool)
        received = np.asarray(data["reception_mask"], dtype=bool)
        identity = np.eye(self.n_uavs, dtype=bool)[None]
        use_direct = identity | (geometric & received)
        use_surrogate = ~identity & geometric & ~received
        blocks = np.where(
            use_direct[..., None],
            direct[:, None, :, :],
            np.where(use_surrogate[..., None], surrogate, 0.0),
        )
        senders = np.arange(self.n_uavs)
        sender_order = np.asarray([
            np.concatenate(([receiver], senders[senders != receiver]))
            for receiver in range(self.n_uavs)
        ])
        ordered = np.take_along_axis(
            blocks, sender_order[None, :, :, None], axis=2
        )
        attention = np.take_along_axis(
            use_direct | use_surrogate, sender_order[None], axis=2
        ).astype(np.float32)
        self.clock += 1
        return ordered.reshape(environments, self.n_uavs, -1), attention

    @staticmethod
    def _age_fraction_metrics(prefix, counts):
        total = int(np.sum(counts))
        denominator = max(total, 1)
        return {
            f"{prefix}_count": total,
            f"{prefix}_age1_fraction": float(counts[0] / denominator),
            f"{prefix}_age2_fraction": float(counts[1] / denominator),
            f"{prefix}_age3_fraction": float(counts[2] / denominator),
            f"{prefix}_age4plus_fraction": float(counts[3] / denominator),
        }

    def _predict_sequence_batch(self, receiver, buffer, pairs):
        if not len(pairs):
            return None
        lengths_np = buffer.lengths[pairs[:, 0], pairs[:, 1]].astype(np.int64)
        features = torch.as_tensor(
            buffer.features[pairs[:, 0], pairs[:, 1]],
            device=self.device,
            dtype=torch.float32,
        )
        gaps = torch.as_tensor(
            buffer.gaps[pairs[:, 0], pairs[:, 1]].astype(np.float32)
            / max(self.args.md_lifetime_max, 1),
            device=self.device,
            dtype=torch.float32,
        )
        lengths = torch.as_tensor(lengths_np, device=self.device, dtype=torch.long)
        predictor = self.predictors[receiver]
        hidden = torch.zeros(
            len(pairs), self.hidden_dim, device=self.device, dtype=torch.float32
        )
        hidden = predictor.update(features[:, 0], gaps[:, 0, None], hidden)
        previous = features[:, 0]
        loss_sum = torch.zeros((), device=self.device)
        squared_position_error = torch.zeros((), device=self.device)
        squared_last_obs_error = torch.zeros((), device=self.device)
        target_count = 0
        x_scale = 0.5 * max(self.args.x_max_gu - self.args.x_min_gu, 1e-6)
        y_scale = 0.5 * max(self.args.y_max_gu - self.args.y_min_gu, 1e-6)
        for timestep in range(1, buffer.max_events):
            active = lengths > timestep
            if not torch.any(active):
                break
            baseline = previous if self.residual_prediction[receiver] else None
            prediction = predictor.predict(hidden, gaps[:, timestep, None], baseline)
            target = features[:, timestep]
            per_feature = nn.functional.smooth_l1_loss(
                prediction, target, reduction="none"
            )
            loss_sum = loss_sum + per_feature[active].sum()
            position_error = (
                ((prediction[:, 0] - target[:, 0]) * x_scale) ** 2
                + ((prediction[:, 1] - target[:, 1]) * y_scale) ** 2
            )
            last_obs_error = (
                ((previous[:, 0] - target[:, 0]) * x_scale) ** 2
                + ((previous[:, 1] - target[:, 1]) * y_scale) ** 2
            )
            squared_position_error += position_error[active].sum()
            squared_last_obs_error += last_obs_error[active].sum()
            target_count += int(torch.count_nonzero(active).item())
            updated = predictor.update(target, gaps[:, timestep, None], hidden)
            hidden = torch.where(active[:, None], updated, hidden)
            previous = torch.where(active[:, None], target, previous)
        if target_count == 0:
            return None
        return {
            "loss": loss_sum / (target_count * self.feature_dim),
            "target_count": target_count,
            "position_sse": squared_position_error,
            "last_obs_sse": squared_last_obs_error,
        }

    @torch.no_grad()
    def _evaluate_pairs(self, receiver, buffer, pairs):
        if not len(pairs):
            return 0.0, 0.0, 0
        self.predictors[receiver].eval()
        position_sse = 0.0
        last_obs_sse = 0.0
        count = 0
        batch_size = int(self.args.md_gru_batch_size)
        for start in range(0, len(pairs), batch_size):
            result = self._predict_sequence_batch(
                receiver, buffer, pairs[start:start + batch_size]
            )
            if result is None:
                continue
            position_sse += float(result["position_sse"].cpu())
            last_obs_sse += float(result["last_obs_sse"].cpu())
            count += result["target_count"]
        return position_sse, last_obs_sse, count

    def _prediction_diagnostics(
        self, receiver, label_counts, prediction_sse, last_obs_sse, sample_count
    ):
        query_counts = self.query_age_counts[receiver].copy()
        diagnostics = {
            **self._age_fraction_metrics("md_prediction_rollout_label", label_counts),
            # Preserve old dashboards during the replay-to-session migration.
            **self._age_fraction_metrics("md_prediction_replay_label", label_counts),
            **self._age_fraction_metrics("md_prediction_rollout_query", query_counts),
            "md_prediction_prequential_samples": int(sample_count),
            "md_prediction_prequential_rmse_m": float(np.sqrt(
                prediction_sse / max(sample_count, 1)
            )),
            "md_last_obs_prequential_rmse_m": float(np.sqrt(
                last_obs_sse / max(sample_count, 1)
            )),
        }
        self.query_age_counts[receiver] = 0
        return diagnostics

    def train_predictors(self):
        metrics = []
        for receiver in range(self.n_uavs):
            buffer = self.session_buffers[receiver]
            pairs = buffer.session_pairs()
            total_targets = buffer.target_count(pairs)
            label_counts = buffer.age_counts(pairs)
            if len(pairs):
                pairs = pairs[self.training_rng.permutation(len(pairs))]
            train_pairs = buffer.take_target_budget(pairs, self.train_samples)
            train_keys = {
                (int(environment), int(session_id))
                for environment, session_id in train_pairs
            }
            remaining_pairs = np.asarray(
                [pair for pair in pairs if tuple(map(int, pair)) not in train_keys],
                dtype=np.int64,
            )
            if remaining_pairs.size == 0:
                remaining_pairs = np.zeros((0, 2), dtype=np.int64)
            validation_pairs = buffer.take_target_budget(
                remaining_pairs, max(64, self.train_samples // 4)
            )
            diagnostic_pairs = validation_pairs if len(validation_pairs) else train_pairs
            pre_sse, pre_last_sse, pre_count = self._evaluate_pairs(
                receiver, buffer, diagnostic_pairs
            )
            diagnostics = self._prediction_diagnostics(
                receiver, label_counts, pre_sse, pre_last_sse, pre_count
            )
            session_count = int(len(pairs))
            if total_targets < self.min_ready_samples or not len(train_pairs):
                metrics.append({
                    "md_prediction_loss": 0.0,
                    "md_prediction_samples": total_targets,
                    "md_prediction_replay_size": total_targets,
                    "md_prediction_rollout_sessions": session_count,
                    "md_prediction_train_samples": 0,
                    "md_prediction_train_sessions": 0,
                    "md_prediction_validation_samples": 0,
                    "md_prediction_position_rmse_m": 0.0,
                    "md_prediction_age_mean": 0.0,
                    "md_prediction_age_ge2_fraction": 0.0,
                    "md_prediction_enabled": float(self.predictor_ready[receiver]),
                    **diagnostics,
                })
                buffer.reset()
                continue

            predictor = self.predictors[receiver]
            optimizer = self.optimizers[receiver]
            predictor.train()
            epoch_losses = []
            position_sse = 0.0
            trained_targets = 0
            batch_size = int(self.args.md_gru_batch_size)
            for epoch in range(int(self.args.md_gru_epochs)):
                epoch_pairs = train_pairs[
                    self.training_rng.permutation(len(train_pairs))
                ]
                for start in range(0, len(epoch_pairs), batch_size):
                    result = self._predict_sequence_batch(
                        receiver, buffer, epoch_pairs[start:start + batch_size]
                    )
                    if result is None:
                        continue
                    optimizer.zero_grad()
                    result["loss"].backward()
                    nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                    optimizer.step()
                    epoch_losses.append(float(result["loss"].detach().cpu()))
                    if epoch == int(self.args.md_gru_epochs) - 1:
                        position_sse += float(result["position_sse"].detach().cpu())
                        trained_targets += result["target_count"]
            predictor.eval()

            validation_sse, validation_last_sse, validation_count = (
                self._evaluate_pairs(receiver, buffer, validation_pairs)
            )
            if validation_count:
                self.predictor_ready[receiver] = validation_sse <= validation_last_sse
            else:
                # Tiny unit/smoke rollouts may not provide a disjoint holdout.
                self.predictor_ready[receiver] = True
            ages = buffer.age_values(train_pairs)
            metrics.append({
                "md_prediction_loss": float(np.mean(epoch_losses)),
                "md_prediction_samples": total_targets,
                "md_prediction_replay_size": total_targets,
                "md_prediction_rollout_sessions": session_count,
                "md_prediction_train_samples": buffer.target_count(train_pairs),
                "md_prediction_train_sessions": int(len(train_pairs)),
                "md_prediction_validation_samples": int(validation_count),
                "md_prediction_position_rmse_m": float(np.sqrt(
                    position_sse / max(trained_targets, 1)
                )),
                "md_prediction_validation_rmse_m": float(np.sqrt(
                    validation_sse / max(validation_count, 1)
                )),
                "md_prediction_validation_last_obs_rmse_m": float(np.sqrt(
                    validation_last_sse / max(validation_count, 1)
                )),
                "md_prediction_age_mean": float(np.mean(ages)) if len(ages) else 0.0,
                "md_prediction_age_ge2_fraction": (
                    float(np.mean(ages >= 2)) if len(ages) else 0.0
                ),
                "md_prediction_enabled": float(self.predictor_ready[receiver]),
                **diagnostics,
            })
            buffer.reset()
        return metrics
