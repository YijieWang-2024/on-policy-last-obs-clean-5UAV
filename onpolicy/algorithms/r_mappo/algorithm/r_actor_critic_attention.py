import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.utils.util import get_shape_from_obs_space


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads=4):
        super(AttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.head_dim = output_dim // n_heads
        assert input_dim % n_heads == 0, "input_dim必须能被num_heads整除"
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.proj = nn.Linear(output_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        # Initialize weights
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0))
        self.query = init_(self.query)
        self.key = init_(self.key)
        self.value = init_(self.value)
        self.proj = init_(self.proj)

    def forward(self, x, mask=None):
        # x: [batch_size, num_agents, input_dim]
        batch_size, num_agents, _ = x.shape
        # [batch_size, num_agents, output_dim]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # Reshape for multi-head attention
        # [batch_size, num_agents, n_heads, head_dim]
        q = q.reshape(batch_size, num_agents, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, num_agents, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, num_agents, self.n_heads, self.head_dim)
        # Transpose for attention calculation
        # [batch_size, n_heads, num_agents, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Calculate attention scores
        # [batch_size, n_heads, num_agents, num_agents]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # Convert mask from [batch_size, num_agents] to [batch_size, 1, 1, num_agents]
            mask_expand = mask.unsqueeze(1).unsqueeze(2)
            mask_expand = mask_expand.expand_as(scores)
            scores = scores.masked_fill(mask_expand == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights
        # [batch_size, n_heads, num_agents, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        # [batch_size, num_agents, n_heads, head_dim] -> [batch_size, num_agents, output_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_agents, -1)

        # Project and normalize
        output = self.proj(attn_output)
        output = self.layer_norm(output + x)

        # If a mask was provided, we need to ensure that masked agents' outputs are zeroed
        if mask is not None:
            # Expand mask to [batch_size, num_agents, 1]
            expanded_mask = mask.unsqueeze(-1)
            # Apply mask to output
            output = output * expanded_mask

        return output


class R_Actor_Attention(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor_Attention, self).__init__()
        self.hidden_size = args.hidden_size
        self.max_UAVs_obs_concat = args.max_UAVs_obs_concat
        self.state_is_k_hops = args.state_is_k_hops
        self.n_UAVs = args.n_UAVs
        assert self.max_UAVs_obs_concat == self.n_UAVs
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self.tpdv = dict(dtype=torch.float32, device=device)

        # Get observation space dimensions
        obs_shape = get_shape_from_obs_space(obs_space)

        # Individual agent observation dimension
        original_feature_size = obs_shape[0]
        if self.max_UAVs_obs_concat > 1:
            # Assuming the last dimension can be divided into max_UAVs_obs_concat parts
            self.individual_obs_dim = original_feature_size // self.max_UAVs_obs_concat
        else:
            self.individual_obs_dim = original_feature_size

        # Encoder for individual agent observations
        self.agent_encoder = MLPBase(args, [self.individual_obs_dim], layer_N=0)

        # Attention layer for inter-agent information exchange
        self.attention = AttentionLayer(self.hidden_size, self.hidden_size*2)

        # Final MLP after attention
        self.mlp_after_attention = MLPBase(args, [self.hidden_size], layer_N=0+1)

        # Action module
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args=args)

        # Register parameters for PyTorch to track
        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False,attention_active_mask=None):

        # Check and convert input types
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if not self.state_is_k_hops:  # local-state（s_{i,t}）或者全局拼接self+所有的state。都不需要active_mask。
            attention_active_mask = None
        if attention_active_mask is not None:
            attention_active_mask = check(attention_active_mask).to(**self.tpdv)

        batch_size = obs.shape[0]

        # Reshape observations to split into individual agent observations
        if self.max_UAVs_obs_concat > 1:
            # [batch_size, obs_dim] -> [batch_size, max_UAVs_obs_concat, individual_obs_dim]
            obs_reshaped = obs.view(batch_size, -1, self.individual_obs_dim)
            agent_features = self.agent_encoder(obs_reshaped)
            # Apply attention mechanism with mask
            # [batch_size, max_UAVs_obs_concat, hidden_size]
            attended_features = self.attention(agent_features, attention_active_mask)

            # 上边的代码使用了self-attention。这里对max_UAVs_obs_concat求了平均。（不平均的话，其实只取第一个观测，即自己的观测就好。）
            if attention_active_mask is not None:
                # Expand mask to match attended_features shape for broadcasting
                expanded_mask = attention_active_mask.unsqueeze(-1)
                # Compute weighted sum (using mask as weights)
                masked_sum = torch.sum(attended_features * expanded_mask, dim=1)
                valid_agents = torch.sum(attention_active_mask, dim=1, keepdim=True).clamp(min=1.0)  # Avoid division by zero
                aggregated_features = masked_sum / valid_agents
            else:
                # Simple mean if no mask provided
                aggregated_features = attended_features.mean(dim=1)
            # # 不平均，只取自己的观测试一下。
            # aggregated_features = attended_features[:, 0]

            # Final MLP processing
            features = self.mlp_after_attention(aggregated_features)

        else:
            # Just process as a single agent if max_UAVs_obs_concat is 1
            features = self.agent_encoder(obs)

        # Action selection
        actions, action_log_probs = self.act(features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None, attention_active_mask=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        if not self.state_is_k_hops:    # local-state（s_{i,t}）或者全局拼接self+所有的state。都不需要active_mask。
            attention_active_mask = None
        if attention_active_mask is not None:
            attention_active_mask = check(attention_active_mask).to(**self.tpdv)

        batch_size = obs.shape[0]

        # Reshape observations to split into individual agent observations
        if self.max_UAVs_obs_concat > 1:
            # [batch_size, obs_dim] -> [batch_size, max_UAVs_obs_concat, individual_obs_dim]
            obs_reshaped = obs.view(batch_size, -1, self.individual_obs_dim)

            agent_features = self.agent_encoder(obs_reshaped)

            # Apply attention mechanism with mask
            attended_features = self.attention(agent_features, attention_active_mask)

            # 上边的代码使用了self-attention。这里对max_UAVs_obs_concat求了平均。（不平均的话，其实只取第一个观测，即自己的观测就好。）
            if attention_active_mask is not None:
                # Expand mask to match attended_features shape for broadcasting
                expanded_mask = attention_active_mask.unsqueeze(-1)
                # Compute weighted sum (using mask as weights)
                masked_sum = torch.sum(attended_features * expanded_mask, dim=1)
                valid_agents = torch.sum(attention_active_mask, dim=1, keepdim=True).clamp(min=1.0)  # Avoid division by zero
                aggregated_features = masked_sum / valid_agents
            else:
                # Simple mean if no mask provided
                aggregated_features = attended_features.mean(dim=1)
            # # 不平均，只取自己的观测试一下。
            # aggregated_features = attended_features[:, 0]

            # Final MLP processing
            features = self.mlp_after_attention(aggregated_features)
        else:
            # Single agent processing
            features = self.agent_encoder(obs)
        # Action evaluation
        action_log_probs, dist_entropy = self.act.evaluate_actions(features, action, available_actions, active_masks=
                                                                active_masks if self._use_policy_active_masks
                                                                else None)

        return action_log_probs, dist_entropy


class R_Critic_Attention(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic_Attention, self).__init__()
        self.hidden_size = args.hidden_size
        self.max_UAVs_obs_concat = args.max_UAVs_obs_concat if hasattr(args, 'max_UAVs_obs_concat') else 1
        self.perform_with_local_state = args.perform_with_local_state
        self.state_is_k_hops = args.state_is_k_hops
        self.n_UAVs = args.n_UAVs
        assert self.max_UAVs_obs_concat == self.n_UAVs
        self._use_orthogonal = args.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)
        # Get centralized observation space dimensions
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        # Individual agent observation dimension
        original_feature_size = cent_obs_shape[0]

        # if self.perform_with_local_state:
        #     if self.max_UAVs_obs_concat > 1:
        #         # Assuming the last dimension can be divided into max_UAVs_obs_concat parts
        #         self.individual_obs_dim = original_feature_size // self.max_UAVs_obs_concat
        #     else:
        #         self.individual_obs_dim = original_feature_size
        # else:
        #     self.individual_obs_dim = original_feature_size // (self.n_UAVs + 1)

        if self.perform_with_local_state:
            self.individual_obs_dim = original_feature_size
        elif self.state_is_k_hops:   #k-hops或者直接拼接self+全局
            self.individual_obs_dim = original_feature_size // self.max_UAVs_obs_concat
        else:
            self.individual_obs_dim = original_feature_size // (self.n_UAVs + 1)

        # Encoder for individual agent observations
        self.agent_encoder = MLPBase(args, [self.individual_obs_dim], layer_N=0)    # 每一个individual_obs_dim到args.hidden_size，带active_func
        # Attention layer for inter-agent information exchange
        self.attention = AttentionLayer(self.hidden_size, self.hidden_size*2)
        # Final MLP after attention
        self.mlp_after_attention = MLPBase(args, [self.hidden_size], layer_N=0+1)
        # Value head
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)

    def forward(self, cent_obs, rnn_states, masks, attention_active_mask=None):
        # Check and convert input types
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # # 如果是用的全局的critic，不需要attention_mask来指示哪些智能体的状态可用了。全部可用。
        # if not self.perform_with_local_state:
        #     attention_active_mask = None

        if not self.state_is_k_hops:    # local-state（s_{i,t}）或者全局拼接self+所有的state。都不需要active_mask。
            attention_active_mask = None

        if attention_active_mask is not None:
            attention_active_mask = check(attention_active_mask).to(**self.tpdv)

        batch_size = cent_obs.shape[0]

        if not self.perform_with_local_state:
            # [batch_size, cent_obs_dim] -> [batch_size, max_UAVs_obs_concat(或者n_UAVs+1), individual_obs_dim]
            obs_reshaped = cent_obs.view(batch_size, -1, self.individual_obs_dim)
            agent_features = self.agent_encoder(obs_reshaped)
            # Apply attention mechanism with agent mask
            attended_features = self.attention(agent_features, attention_active_mask)
            # 上边的代码使用了self-attention。这里对max_UAVs_obs_concat求了平均。（不平均的话，其实只取第一个观测，即自己的观测就好。）
            if attention_active_mask is not None:
                # Expand mask to match attended_features shape for broadcasting
                expanded_mask = attention_active_mask.unsqueeze(-1)
                # Compute weighted sum (using mask as weights)
                masked_sum = torch.sum(attended_features * expanded_mask, dim=1)
                valid_agents = torch.sum(attention_active_mask, dim=1, keepdim=True).clamp(
                    min=1.0)  # Avoid division by zero
                aggregated_features = masked_sum / valid_agents
            else:
                # Simple mean if no mask provided
                aggregated_features = attended_features.mean(dim=1)
            # # 不平均，只取自己的观测试一下。
            # aggregated_features = attended_features[:, 0]

            # Final MLP processing
            features = self.mlp_after_attention(aggregated_features)
        else:
            # Single agent processing
            features = self.agent_encoder(cent_obs)

        # if self.perform_with_local_state or self.max_UAVs_obs_concat > 1:
        #     # [batch_size, cent_obs_dim] -> [batch_size, max_UAVs_obs_concat, individual_obs_dim]
        #     obs_reshaped = cent_obs.view(batch_size, -1, self.individual_obs_dim)
        #
        #     agent_features = self.agent_encoder(obs_reshaped)
        #
        #     # Apply attention mechanism with agent mask
        #     attended_features = self.attention(agent_features, attention_active_mask)
        #
        #     # 上边的代码使用了self-attention。这里对max_UAVs_obs_concat求了平均。（不平均的话，其实只取第一个观测，即自己的观测就好。）
        #     if attention_active_mask is not None:
        #         # Expand mask to match attended_features shape for broadcasting
        #         expanded_mask = attention_active_mask.unsqueeze(-1)
        #         # Compute weighted sum (using mask as weights)
        #         masked_sum = torch.sum(attended_features * expanded_mask, dim=1)
        #         valid_agents = torch.sum(attention_active_mask, dim=1, keepdim=True).clamp(min=1.0)  # Avoid division by zero
        #         aggregated_features = masked_sum / valid_agents
        #     else:
        #         # Simple mean if no mask provided
        #         aggregated_features = attended_features.mean(dim=1)
        #     # # 不平均，只取自己的观测试一下。
        #     # aggregated_features = attended_features[:, 0]
        #
        #     # Final MLP processing
        #     features = self.mlp_after_attention(aggregated_features)
        # else:
        #     # Single agent processing
        #     features = self.agent_encoder(cent_obs)

        # Value prediction
        values = self.v_out(features)

        return values, rnn_states
