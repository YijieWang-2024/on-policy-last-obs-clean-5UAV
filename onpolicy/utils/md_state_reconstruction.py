import numpy as np
import torch
import torch.nn as nn

from onpolicy.utils.md_roster import (
    order_md_candidates,
    resolve_distance_only_user_sort,
)


class MDGRUPredictor(nn.Module):
    """A single mobility predictor shared by all UAVs and MD identities."""

    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRUCell(feature_dim + 1, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def update(self, feature, age, hidden):
        return self.gru(torch.cat((feature, age), dim=-1), hidden)

    def predict(self, hidden, age):
        return self.head(torch.cat((hidden, age), dim=-1))


class _Reservoir:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.items = []
        self.seen = 0
        self.copied = 0
        self.last_seen = 0
        self.last_copied = 0

    def plan_batch(self, count):
        """Choose retained stream items before copying their GPU contexts."""
        count = int(count)
        if count <= 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty

        start_seen = self.seen
        free = max(self.capacity - len(self.items), 0)
        appended = min(free, count)
        sources = np.arange(appended, dtype=np.int64)
        destinations = np.arange(
            len(self.items), len(self.items) + appended, dtype=np.int64
        )
        self.items.extend([None] * appended)

        remaining = count - appended
        if remaining:
            stream_positions = np.arange(
                start_seen + appended + 1,
                start_seen + count + 1,
                dtype=np.int64,
            )
            draws = self.rng.integers(0, stream_positions)
            accepted = draws < self.capacity
            sources = np.concatenate((
                sources,
                np.arange(appended, count, dtype=np.int64)[accepted],
            ))
            destinations = np.concatenate((destinations, draws[accepted]))
        self.seen += count

        if not len(destinations):
            empty = np.empty(0, dtype=np.int64)
            return empty, empty
        # A destination may be replaced repeatedly within one incoming batch.
        # Only its final occupant must cross the device boundary.
        _, reverse_indices = np.unique(
            destinations[::-1], return_index=True
        )
        keep = len(destinations) - 1 - reverse_indices
        return sources[keep], destinations[keep]

    def commit_batch(
        self, destinations, receiver, context_hidden, context_input,
        prediction_age, targets,
    ):
        self.copied += len(destinations)
        for index, destination in enumerate(destinations):
            self.items[int(destination)] = (
                receiver,
                context_hidden[index].copy(),
                context_input[index].copy(),
                prediction_age[index].copy(),
                targets[index].copy(),
            )

    def take(self):
        items, self.items = self.items, []
        self.last_seen = self.seen
        self.last_copied = self.copied
        self.seen = 0
        self.copied = 0
        return items


class _MemoryBank:
    def __init__(self, environments, capacity, feature_dim, hidden_dim, device):
        self.environments = environments
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.ids = np.full((environments, capacity), -1, dtype=np.int64)
        self.age = np.zeros((environments, capacity), dtype=np.int32)
        self.lifetime = np.zeros((environments, capacity), dtype=np.int32)
        self.last_feature = np.zeros((environments, capacity, feature_dim), dtype=np.float32)
        self.estimate = np.zeros_like(self.last_feature)
        self.tasks = np.zeros((environments, capacity, 3), dtype=np.float32)
        self.task_valid = np.zeros((environments, capacity), dtype=bool)
        self.hidden = torch.zeros(
            environments, capacity, hidden_dim, device=device, dtype=torch.float32
        )
        self.context_hidden = torch.zeros_like(self.hidden)
        self.context_input = torch.zeros(
            environments, capacity, feature_dim + 1, device=device, dtype=torch.float32
        )

    def reset(self, environment_mask=None):
        if environment_mask is None:
            environment_mask = np.ones(self.environments, dtype=bool)
        self.ids[environment_mask] = -1
        self.age[environment_mask] = 0
        self.lifetime[environment_mask] = 0
        self.last_feature[environment_mask] = 0
        self.estimate[environment_mask] = 0
        self.tasks[environment_mask] = 0
        self.task_valid[environment_mask] = False
        mask = torch.as_tensor(environment_mask, device=self.device)
        self.hidden[mask] = 0
        self.context_hidden[mask] = 0
        self.context_input[mask] = 0

    def find(self, environment, session_id):
        matches = np.flatnonzero(self.ids[environment] == session_id)
        return int(matches[0]) if len(matches) else None

    def allocate(self, environment, session_id):
        free = np.flatnonzero(self.ids[environment] == -1)
        if not len(free):
            raise RuntimeError("MD memory capacity was exceeded")
        slot = int(free[0])
        self.ids[environment, slot] = session_id
        self.age[environment, slot] = 0
        self.lifetime[environment, slot] = 0
        self.hidden[environment, slot] = 0
        self.context_hidden[environment, slot] = 0
        self.context_input[environment, slot] = 0
        return slot

    def remove_expired(self, environment, protected_ids=()):
        expired = (self.ids[environment] != -1) & (self.lifetime[environment] <= 0)
        if protected_ids:
            expired &= ~np.isin(self.ids[environment], list(protected_ids))
        if not np.any(expired):
            return
        self.ids[environment, expired] = -1
        self.age[environment, expired] = 0
        self.lifetime[environment, expired] = 0
        self.last_feature[environment, expired] = 0
        self.estimate[environment, expired] = 0
        self.tasks[environment, expired] = 0
        self.task_valid[environment, expired] = False
        mask = torch.as_tensor(expired, device=self.device)
        self.hidden[environment, mask] = 0
        self.context_hidden[environment, mask] = 0
        self.context_input[environment, mask] = 0


class MDStateReconstructor:
    """Causal receiver-ID memory and optional MD-GRU Type-S reconstruction."""

    feature_dim = 7  # x, y, speed, sin/cos(direction), sin/cos(reference direction)

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.mode = args.state_reconstruction
        self.n_uavs = args.n_UAVs
        self.capacity = args.n_GUs
        self.packet_capacity = args.max_GUs_in_range
        self.hidden_dim = args.md_gru_hidden_dim
        self.distance_only_user_sort = resolve_distance_only_user_sort(args)
        self.metadata = bool(getattr(args, "critic_md_metadata", False) or self.mode != "zero")
        self.predictor = (
            MDGRUPredictor(self.feature_dim, self.hidden_dim).to(device)
            if self.mode == "md_gru" else None
        )
        # Public one-element container retained for simple checkpoint/test
        # introspection; there is exactly one shared parameter set.
        self.predictors = (
            nn.ModuleList([self.predictor]) if self.predictor is not None else None
        )
        self.optimizer = (
            torch.optim.Adam(self.predictor.parameters(), lr=args.md_gru_lr)
            if self.predictor is not None else None
        )
        self.reservoir = _Reservoir(args.md_gru_max_samples, args.seed + 7919)
        self.training_rng = np.random.default_rng(args.seed + 15485863)
        # Do not expose a randomly initialized mobility head to the critic.
        # The last observed persistent record is used until one supervised
        # predictor update has completed.
        self.predictor_ready = self.predictor is None
        self.banks = None
        self.environments = None
        self.last_data = None

    def checkpoint_state(self):
        """Return the complete trainable shared-predictor state."""
        if self.predictor is None:
            raise RuntimeError("only md_gru reconstruction has predictor state")
        return {
            "model": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "predictor_ready": self.predictor_ready,
        }

    def load_checkpoint_state(self, checkpoint):
        """Load current checkpoints and the legacy model-only representation."""
        if self.predictor is None:
            raise RuntimeError("only md_gru reconstruction has predictor state")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.predictor.load_state_dict(checkpoint["model"])
            optimizer_state = checkpoint.get("optimizer")
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)
            self.predictor_ready = checkpoint.get("predictor_ready", True)
        else:
            self.predictor.load_state_dict(checkpoint)
            self.predictor_ready = True
        self.predictor.eval()

    def _ensure_banks(self, environments):
        if self.banks is not None and self.environments == environments:
            return
        self.environments = environments
        self.banks = [
            _MemoryBank(
                environments, self.capacity, self.feature_dim,
                self.hidden_dim, self.device,
            )
            for _ in range(self.n_uavs)
        ]

    def reset(self, environment_mask=None):
        if self.banks is None:
            return
        for bank in self.banks:
            bank.reset(environment_mask)

    def _encode_feature(self, record):
        x_span = max(self.args.x_max_gu - self.args.x_min_gu, 1e-6)
        y_span = max(self.args.y_max_gu - self.args.y_min_gu, 1e-6)
        speed_bound = max(1.3 * self.args.mean_velocity, 1e-6)
        return np.asarray([
            2 * (record[0] - self.args.x_min_gu) / x_span - 1,
            2 * (record[1] - self.args.y_min_gu) / y_span - 1,
            2 * record[2] / speed_bound - 1,
            np.sin(record[3]), np.cos(record[3]),
            np.sin(record[4]), np.cos(record[4]),
        ], dtype=np.float32)

    def _decode_feature(self, feature):
        feature = np.asarray(feature, dtype=np.float64)
        x_span = max(self.args.x_max_gu - self.args.x_min_gu, 1e-6)
        y_span = max(self.args.y_max_gu - self.args.y_min_gu, 1e-6)
        speed_bound = max(1.3 * self.args.mean_velocity, 1e-6)
        x = self.args.x_min_gu + 0.5 * (np.clip(feature[0], -1, 1) + 1) * x_span
        y = self.args.y_min_gu + 0.5 * (np.clip(feature[1], -1, 1) + 1) * y_span
        speed = 0.5 * (np.clip(feature[2], -1, 1) + 1) * speed_bound
        direction = np.arctan2(feature[3], feature[4]) % (2 * np.pi)
        reference = np.arctan2(feature[5], feature[6]) % (2 * np.pi)
        return np.asarray((x, y, speed, direction, reference), dtype=np.float64)

    def _age_feature(self, age):
        return np.asarray([age / max(self.args.md_lifetime_max, 1)], dtype=np.float32)

    @staticmethod
    def _observed_records(data, environment, receiver):
        observed = {}
        n_uavs = data["record_features"].shape[1]
        sources = [receiver] + [
            sender for sender in range(n_uavs)
            if sender != receiver
            and data["geometric_mask"][environment, receiver, sender]
            and data["reception_mask"][environment, receiver, sender]
        ]
        for sender in sources:
            valid_slots = np.flatnonzero(data["record_valid"][environment, sender])
            for slot in valid_slots:
                session_id = int(data["session_ids"][environment, sender, slot])
                record = data["record_features"][environment, sender, slot]
                if session_id in observed:
                    common = np.r_[record[:6], record[7:10]]
                    previous = np.r_[observed[session_id][:6], observed[session_id][7:10]]
                    if not np.allclose(common, previous, rtol=1e-6, atol=1e-6):
                        raise AssertionError("duplicate MD records disagree across Type-S sources")
                    continue
                observed[session_id] = record
        return observed

    def _add_samples(self, receiver, bank, environments, slots, ages, targets):
        if not slots:
            return
        selected, destinations = self.reservoir.plan_batch(len(slots))
        if not len(selected):
            return
        environments = np.asarray(environments, dtype=np.int64)[selected]
        slots = np.asarray(slots, dtype=np.int64)[selected]
        environment_index = torch.as_tensor(
            environments, device=self.device, dtype=torch.long
        )
        slot_index = torch.as_tensor(slots, device=self.device, dtype=torch.long)
        context = torch.cat((
            bank.context_hidden[environment_index, slot_index],
            bank.context_input[environment_index, slot_index],
        ), dim=-1).detach().cpu().numpy()
        self.reservoir.commit_batch(
            destinations,
            receiver,
            context[:, :self.hidden_dim],
            context[:, self.hidden_dim:],
            np.asarray(
                [self._age_feature(ages[index]) for index in selected],
                dtype=np.float32,
            ),
            np.asarray([targets[index] for index in selected], dtype=np.float32),
        )

    @torch.no_grad()
    def _update_bank(self, receiver, data, collect_samples):
        bank = self.banks[receiver]
        predictor = self.predictor
        observed_by_environment = [
            self._observed_records(data, environment, receiver)
            for environment in range(self.environments)
        ]

        # Advance all active identities in one vectorized pass. In MD-GRU mode
        # this is one prediction call per receiver, rather than one tiny GPU
        # kernel per environment/session pair.
        bank.task_valid[:] = False
        active = bank.ids != -1
        bank.age[active] += 1
        bank.lifetime[active] -= 1
        for environment in range(self.environments):
            bank.remove_expired(
                environment, protected_ids=set(observed_by_environment[environment])
            )
        active = bank.ids != -1
        active_environments, active_slots = np.nonzero(active)
        if len(active_slots):
            if predictor is None or not self.predictor_ready:
                bank.estimate[active_environments, active_slots] = bank.last_feature[
                    active_environments, active_slots
                ]
            else:
                active_environment_index = torch.as_tensor(
                    active_environments, device=self.device, dtype=torch.long
                )
                active_slot_index = torch.as_tensor(
                    active_slots, device=self.device, dtype=torch.long
                )
                age = torch.as_tensor(
                    bank.age[active_environments, active_slots, None]
                    / max(self.args.md_lifetime_max, 1),
                    device=self.device, dtype=torch.float32,
                )
                prediction = predictor.predict(
                    bank.hidden[active_environment_index, active_slot_index], age
                )
                bank.estimate[active_environments, active_slots] = (
                    prediction.cpu().numpy()
                )

        update_environments = []
        update_slots = []
        update_features = []
        update_ages = []
        sample_environments = []
        sample_slots = []
        sample_ages = []
        sample_targets = []
        for environment in range(self.environments):
            observed = observed_by_environment[environment]
            for session_id, record in observed.items():
                feature = self._encode_feature(record)
                slot = bank.find(environment, session_id)
                is_new = slot is None
                if is_new:
                    slot = bank.allocate(environment, session_id)
                    update_age = 0
                else:
                    update_age = int(bank.age[environment, slot])
                    if predictor is not None and collect_samples:
                        sample_environments.append(environment)
                        sample_slots.append(slot)
                        sample_ages.append(update_age)
                        sample_targets.append(feature)
                update_environments.append(environment)
                update_slots.append(slot)
                update_features.append(feature)
                update_ages.append(update_age)
                bank.last_feature[environment, slot] = feature
                bank.estimate[environment, slot] = feature
                bank.age[environment, slot] = 0
                bank.lifetime[environment, slot] = int(round(record[5]))
                bank.tasks[environment, slot] = record[7:10]
                bank.task_valid[environment, slot] = True

        # Gather all re-observation contexts before this slot overwrites them.
        # This reduces GPU-to-CPU synchronization from once per MD to once per
        # receiver and environment slot.
        self._add_samples(
            receiver, bank, sample_environments, sample_slots,
            sample_ages, sample_targets,
        )

        if update_slots:
            update_environments = np.asarray(update_environments, dtype=np.int64)
            update_slots = np.asarray(update_slots, dtype=np.int64)
            update_features = np.asarray(update_features, dtype=np.float32)
            normalized_ages = np.asarray(
                [self._age_feature(age) for age in update_ages], dtype=np.float32
            )
            feature_tensor = torch.as_tensor(
                update_features, device=self.device, dtype=torch.float32
            )
            age_tensor = torch.as_tensor(
                normalized_ages, device=self.device, dtype=torch.float32
            )
            update_input = torch.cat((feature_tensor, age_tensor), dim=-1)
            update_environment_index = torch.as_tensor(
                update_environments, device=self.device, dtype=torch.long
            )
            update_slot_index = torch.as_tensor(
                update_slots, device=self.device, dtype=torch.long
            )
            old_hidden = bank.hidden[
                update_environment_index, update_slot_index
            ].clone()
            bank.context_hidden[
                update_environment_index, update_slot_index
            ] = old_hidden
            bank.context_input[
                update_environment_index, update_slot_index
            ] = update_input
            if predictor is not None:
                bank.hidden[
                    update_environment_index, update_slot_index
                ] = predictor.gru(update_input, old_hidden)

    def _channel_gain(self, sender_position, md_position):
        horizontal = np.linalg.norm(sender_position - md_position)
        vertical = self.args.H_UAV - self.args.H_GU
        distance = max(np.hypot(horizontal, vertical), 1e-10)
        theta = 180 / np.pi * np.arcsin(np.clip(vertical / distance, -1, 1))
        p_los = 1 / (1 + 9.61 * np.exp(-0.16 * (theta - 9.61)))
        log_term = 20 * np.log10(4 * np.pi * 2e9 / 3e8)
        path_loss = (
            p_los * (log_term + 1 + 20 * np.log10(distance))
            + (1 - p_los) * (log_term + 20 + 20 * np.log10(distance))
        )
        return 10 ** (-path_loss / 10)

    def _direct_block(self, data, environment, sender):
        pieces = [
            data["critic_prefix"][environment, sender],
            data["record_features"][environment, sender].reshape(-1),
        ]
        if self.metadata:
            valid = data["record_valid"][environment, sender].astype(np.float64)
            metadata = np.stack((valid, valid, np.zeros_like(valid)), axis=-1)
            pieces.append(metadata.reshape(-1))
        return np.concatenate(pieces)

    def _surrogate_prefix(
        self, data, environment, receiver, sender, visible_count
    ):
        values = []
        if self.args.ob_state_with_timestep:
            # Time is locally known even when the sender packet is lost.
            values.append(data["critic_prefix"][environment, receiver, :1])
        if self.args.ob_state_with_id:
            identity = np.zeros(self.n_uavs, dtype=np.float64)
            identity[sender] = 1
            values.append(identity)
        values.append(data["uav_positions"][environment, sender, :2])
        layout_dim = 8 if getattr(self.args, "episode_layout_context", False) else 0
        if layout_dim:
            layout_start = (
                int(self.args.ob_state_with_timestep)
                + (self.n_uavs if self.args.ob_state_with_id else 0)
                + 2
            )
            values.append(data["critic_prefix"][
                environment, receiver, layout_start:layout_start + layout_dim
            ])
        values.append(np.asarray([visible_count], dtype=np.float64))
        return np.concatenate(values)

    def _surrogate_block(self, data, environment, receiver, sender):
        bank = self.banks[receiver]
        slots = np.flatnonzero(bank.ids[environment] != -1)
        sender_position = data["uav_positions"][environment, sender, :2]
        candidates = []
        for slot in slots:
            decoded = self._decode_feature(bank.estimate[environment, slot])
            distance = float(np.linalg.norm(decoded[:2] - sender_position))
            if distance > self.args.Cover_R:
                continue
            task_valid = bool(bank.task_valid[environment, slot])
            if task_valid:
                task = bank.tasks[environment, slot]
                can_finish = task[1] / self.args.F_n <= task[2]
            else:
                can_finish = False
            candidates.append((
                distance, int(bank.ids[environment, slot]), slot, decoded,
                task_valid, can_finish,
            ))
        if candidates:
            order = order_md_candidates(
                [item[0] for item in candidates],
                [item[1] for item in candidates],
                [item[5] for item in candidates],
                [item[4] for item in candidates],
                distance_only=self.distance_only_user_sort,
            )
            candidates = [candidates[index] for index in order]
        visible_count = len(candidates)
        records = np.zeros((self.packet_capacity, 10), dtype=np.float64)
        metadata = np.zeros((self.packet_capacity, 3), dtype=np.float64)
        placeholder = np.asarray([
            0.5 * (self.args.D_min + self.args.D_max),
            0.5 * (self.args.C_min + self.args.C_max),
            0.5 * (self.args.delay_min + self.args.delay_max),
        ])
        for record_slot, (_, _, memory_slot, decoded, _, _) in enumerate(
            candidates[:self.packet_capacity]
        ):
            records[record_slot, :5] = decoded
            records[record_slot, 5] = bank.lifetime[environment, memory_slot]
            records[record_slot, 6] = self._channel_gain(sender_position, decoded[:2])
            task_valid = bool(bank.task_valid[environment, memory_slot])
            records[record_slot, 7:10] = (
                bank.tasks[environment, memory_slot] if task_valid else placeholder
            )
            metadata[record_slot] = (
                1.0,
                float(task_valid),
                bank.age[environment, memory_slot] / max(self.args.md_lifetime_max, 1),
            )
        pieces = [
            self._surrogate_prefix(
                data, environment, receiver, sender, visible_count
            ),
            records.reshape(-1),
        ]
        if self.metadata:
            pieces.append(metadata.reshape(-1))
        return np.concatenate(pieces)

    def reconstruct(self, data, reset_environments=None, collect_samples=True):
        environments = data["record_features"].shape[0]
        self._ensure_banks(environments)
        if reset_environments is not None:
            self.reset(np.asarray(reset_environments, dtype=bool))
        self.last_data = data
        for receiver in range(self.n_uavs):
            self._update_bank(receiver, data, collect_samples)

        block_dim = (
            data["critic_prefix"].shape[-1]
            + self.packet_capacity * 10
            + (self.packet_capacity * 3 if self.metadata else 0)
        )
        states = np.zeros(
            (environments, self.n_uavs, self.n_uavs * block_dim), dtype=np.float64
        )
        attention = np.zeros(
            (environments, self.n_uavs, self.n_uavs), dtype=np.float32
        )
        all_senders = np.arange(self.n_uavs)
        for environment in range(environments):
            for receiver in range(self.n_uavs):
                sender_order = np.concatenate((
                    [receiver], all_senders[all_senders != receiver]
                ))
                for block_slot, sender in enumerate(sender_order):
                    geometric = bool(data["geometric_mask"][environment, receiver, sender])
                    received = bool(data["reception_mask"][environment, receiver, sender])
                    if sender == receiver or (geometric and received):
                        block = self._direct_block(data, environment, sender)
                    elif geometric:
                        block = self._surrogate_block(
                            data, environment, receiver, sender
                        )
                    else:
                        continue
                    start = block_slot * block_dim
                    states[environment, receiver, start:start + block_dim] = block
                    attention[environment, receiver, block_slot] = 1
        return states, attention

    def train_predictors(self):
        if self.predictor is None:
            return [
                {
                    "md_prediction_loss": 0.0,
                    "md_prediction_samples": 0,
                    "md_prediction_position_rmse_m": 0.0,
                    "md_prediction_candidates_global": self.reservoir.last_seen,
                    "md_prediction_context_rows_copied": self.reservoir.last_copied,
                    "md_prediction_reservoir_fraction": 0.0,
                    "md_prediction_age_mean": 0.0,
                    "md_prediction_age_ge2_fraction": 0.0,
                }
                for _ in range(self.n_uavs)
            ]
        samples = self.reservoir.take()
        if not samples:
            return [
                {
                    "md_prediction_loss": 0.0,
                    "md_prediction_samples": 0,
                    "md_prediction_position_rmse_m": 0.0,
                    "md_prediction_candidates_global": self.reservoir.last_seen,
                    "md_prediction_context_rows_copied": self.reservoir.last_copied,
                    "md_prediction_reservoir_fraction": 0.0,
                    "md_prediction_age_mean": 0.0,
                    "md_prediction_age_ge2_fraction": 0.0,
                }
                for _ in range(self.n_uavs)
            ]
        receivers, context_hidden, context_input, prediction_age, targets = map(
            np.asarray, zip(*samples)
        )
        sample_counts = np.bincount(
            receivers.astype(np.int64), minlength=self.n_uavs
        )
        ages = prediction_age[:, 0] * max(self.args.md_lifetime_max, 1)
        count = len(samples)
        epoch_losses = []
        position_squared_error = 0.0
        position_sample_count = 0
        x_scale = 0.5 * max(self.args.x_max_gu - self.args.x_min_gu, 1e-6)
        y_scale = 0.5 * max(self.args.y_max_gu - self.args.y_min_gu, 1e-6)
        self.predictor.train()
        for epoch in range(self.args.md_gru_epochs):
            order = self.training_rng.permutation(count)
            for start in range(0, count, self.args.md_gru_batch_size):
                indices = order[start:start + self.args.md_gru_batch_size]
                hidden = torch.as_tensor(
                    context_hidden[indices], device=self.device, dtype=torch.float32
                )
                update_input = torch.as_tensor(
                    context_input[indices], device=self.device, dtype=torch.float32
                )
                age = torch.as_tensor(
                    prediction_age[indices], device=self.device, dtype=torch.float32
                )
                target = torch.as_tensor(
                    targets[indices], device=self.device, dtype=torch.float32
                )
                updated_hidden = self.predictor.gru(update_input, hidden)
                prediction = self.predictor.predict(updated_hidden, age)
                loss = torch.mean((prediction - target) ** 2)
                if epoch == self.args.md_gru_epochs - 1:
                    position_error = (
                        ((prediction[:, 0] - target[:, 0]) * x_scale) ** 2
                        + ((prediction[:, 1] - target[:, 1]) * y_scale) ** 2
                    )
                    position_squared_error += float(
                        torch.sum(position_error).detach().cpu()
                    )
                    position_sample_count += len(indices)
                self.optimizer.zero_grad()
                (self.args.md_prediction_loss_coef * loss).backward()
                nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
                self.optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
        self.predictor.eval()
        self.predictor_ready = True
        common_loss = float(np.mean(epoch_losses))
        position_rmse = float(np.sqrt(
            position_squared_error / max(position_sample_count, 1)
        ))
        metrics = []
        for receiver in range(self.n_uavs):
            receiver_ages = ages[receivers.astype(np.int64) == receiver]
            metrics.append({
                "md_prediction_loss": common_loss,
                "md_prediction_samples": sample_counts[receiver],
                "md_prediction_samples_global": count,
                "md_prediction_candidates_global": self.reservoir.last_seen,
                "md_prediction_context_rows_copied": self.reservoir.last_copied,
                "md_prediction_reservoir_fraction": (
                    count / max(self.reservoir.last_seen, 1)
                ),
                "md_prediction_position_rmse_m": position_rmse,
                "md_prediction_age_mean": (
                    float(np.mean(receiver_ages)) if len(receiver_ages) else 0.0
                ),
                "md_prediction_age_ge2_fraction": (
                    float(np.mean(receiver_ages >= 2.0))
                    if len(receiver_ages) else 0.0
                ),
            })
        return metrics
