"""Shared reference policy: one implementation, identical weights, three adapters.

Collection-only experiments (A/B) use a *fixed deterministic policy* (argmax
over logits of a shared-trunk MLP) whose weights are generated once from a
recorded seed and loaded — tensor-identical — into each framework's native
policy object:

- zeroRL:  ``ActorCriticAgent`` (its default agent; keys match 1:1)
- SB3:     ``SharedTrunkPolicy`` (see collection.py) via head/trunk mapping
- TorchRL: custom shared-trunk module (see collection/ppo modules)

Protocol C/D training runs start from this same initial state dict in all
three frameworks (architecture + init aligned).

Adapter overhead is part of what the benchmark measures; every mapping is
asserted tensor-equal after loading so no framework trains/collects on
different weights than the others.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from zerorl.helpers.agent import eval_action
from zerorl.helpers.factory import ActorCriticAgent

OBS_DIM = 8          # LunarLander-v3
ACT_DIM = 4           # LunarLander-v3 (discrete)
HIDDEN = 64           # shared-trunk width (matches zeroRL / SB3 net_arch)
REFERENCE_SEED = 4242

# Expected zeroRL state-dict key layout (shared trunk + heads).
REFERENCE_KEYS = (
    "extract_layer.0.weight",
    "extract_layer.0.bias",
    "extract_layer.2.weight",
    "extract_layer.2.bias",
    "actor.weight",
    "actor.bias",
    "critic.weight",
    "critic.bias",
)


def build_reference_state_dict(seed: int = REFERENCE_SEED) -> dict[str, torch.Tensor]:
    """Deterministic reference weights via zeroRL's own agent init.

    Uses ``ActorCriticAgent`` so the init scheme (ortho gains √2 trunk /
    0.01 actor / 1.0 critic, zero biases) and key layout are exactly zeroRL's
    defaults — the other frameworks map these tensors onto their modules.
    """
    torch.manual_seed(seed)
    agent = ActorCriticAgent(OBS_DIM, ACT_DIM, is_discrete=True, hidden_dim=HIDDEN)
    sd = {k: v.detach().clone() for k, v in agent.state_dict().items()}
    missing = [k for k in REFERENCE_KEYS if k not in sd]
    if missing:
        raise RuntimeError(f"reference state dict missing keys: {missing}")
    return sd


def state_dict_hash(sd: dict[str, torch.Tensor]) -> str:
    """Short deterministic fingerprint for metadata (order-independent)."""
    h = torch.zeros(1)
    for key in sorted(sd):
        h += sd[key].double().sum()
    return f"{float(h):.6e}"


# ---------------------------------------------------------------------------
# zeroRL adapter
# ---------------------------------------------------------------------------

def load_into_zerorl(agent: ActorCriticAgent,
                     sd: dict[str, torch.Tensor]) -> None:
    agent.load_state_dict(sd)
    _assert_equal(agent.state_dict(), sd, "zeroRL agent")


class DeterministicActorCriticAgent(ActorCriticAgent):
    """zeroRL agent whose ``get_action`` is deterministic (argmax).

    Keeps the full official output contract (action/log_prob/entropy/value)
    so ``rollout_phase``/``buffer.insert`` work unchanged; log-prob/entropy
    are evaluated for the *taken* (argmax) action under the policy
    distribution, exactly as the stochastic path would for that action.
    """

    @torch.no_grad()
    def get_action(self, state: torch.Tensor, action: torch.Tensor | None = None,
                   **kwargs: Any) -> dict[str, torch.Tensor]:
        logits, value = self.forward(state, **kwargs)
        dist = self.build_distribution(logits)
        greedy = logits.argmax(dim=-1)
        log_prob, entropy = eval_action(dist, greedy)
        return {
            "action": greedy,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value.squeeze(-1),
        }


def build_deterministic_zerorl_agent(
    sd: dict[str, torch.Tensor] | None = None,
) -> DeterministicActorCriticAgent:
    agent = DeterministicActorCriticAgent(OBS_DIM, ACT_DIM, is_discrete=True,
                                          hidden_dim=HIDDEN)
    load_into_zerorl(agent, sd if sd is not None else build_reference_state_dict())
    return agent


# ---------------------------------------------------------------------------
# SB3 adapter
# ---------------------------------------------------------------------------

def shared_trunk_policy_cls() -> type[nn.Module]:
    """Build SB3's SharedTrunkPolicy *class* (lazy sb3 import; used as ``policy=``).

    Shared-trunk contract: ``_SharedOnceMlpExtractor`` guarantees the trunk
    module executes ONCE per actor-critic pass (SB3's stock
    ``MlpExtractor.forward`` calls ``forward_actor`` then ``forward_critic``,
    which would run the identity-shared ``value_net is policy_net`` module
    twice per observation).

    Since SB3 1.8 ``net_arch=[64, 64]`` builds TWO separate trunks. This
    subclass re-unifies the value trunk with the policy trunk after
    ``_build`` and rebuilds the optimizer from the de-duplicated parameters,
    restoring the shared-trunk topology every framework in this benchmark
    uses (verified parameter count == zeroRL's 5061 on LunarLander-v3).
    """
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import MlpExtractor

    class _SharedOnceMlpExtractor(MlpExtractor):  # type: ignore[misc]
        """Run the shared trunk once per actor-critic pass.

        With ``value_net is policy_net`` (enforced by SharedTrunkPolicy)
        both latents are the same tensor by construction, so the second
        execution would be pure duplicated compute.
        """

        def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            latent_pi = self.forward_actor(features)
            if self.value_net is self.policy_net:
                return latent_pi, latent_pi
            return latent_pi, self.forward_critic(features)

    class SharedTrunkPolicy(ActorCriticPolicy):  # type: ignore[misc]
        def _build(self, lr_schedule_arg: Any) -> None:
            super()._build(lr_schedule_arg)
            if self.mlp_extractor.value_net is not self.mlp_extractor.policy_net:
                self.mlp_extractor.value_net = self.mlp_extractor.policy_net
                self.optimizer = self.optimizer_class(  # type: ignore[attr-defined]
                    self.parameters(),
                    lr=lr_schedule_arg(1),
                    **self.optimizer_kwargs,  # type: ignore[attr-defined]
                )
            if self.mlp_extractor.value_net is self.mlp_extractor.policy_net:
                self.mlp_extractor.__class__ = _SharedOnceMlpExtractor

        def forward_collection(
            self, obs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """One trunk pass returning (actions, values, log_prob, entropy).

            Experiment A contract: every framework computes all four fields
            per transition (zeroRL's ``get_action`` does; SB3's ``predict``
            does NOT — it stops after dist + action). Mirrors
            ``ActorCriticPolicy.forward`` (sb3 policies.py:636-658) plus
            ``distribution.entropy()``; actions are the deterministic mode so
            trajectories match ``predict(deterministic=True)`` exactly.
            """
            features = self.extract_features(obs)
            if self.share_features_extractor:
                latent_pi, latent_vf = self.mlp_extractor(features)
            else:
                pi_features, vf_features = features
                latent_pi = self.mlp_extractor.forward_actor(pi_features)
                latent_vf = self.mlp_extractor.forward_critic(vf_features)
            values = self.value_net(latent_vf)
            distribution = self._get_action_dist_from_latent(latent_pi)
            actions = distribution.get_actions(deterministic=True)
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()
            actions = actions.reshape((-1, *self.action_space.shape))
            return actions, values, log_prob, entropy

    return SharedTrunkPolicy  # type: ignore[return-value]


def build_sb3_shared_policy(
    observation_space: Any,
    action_space: Any,
    *,
    net_arch: tuple[int, ...] = (HIDDEN, HIDDEN),
    lr_schedule: Any = None,
) -> nn.Module:
    """Instantiate SharedTrunkPolicy for the collection experiments."""
    if lr_schedule is None:
        lr_schedule = lambda _: 3e-4  # noqa: E731 - matches PPO default

    SharedTrunkPolicy = shared_trunk_policy_cls()
    policy = SharedTrunkPolicy(  # type: ignore[call-arg]
        observation_space,
        action_space,
        net_arch=list(net_arch),
        lr_schedule=lr_schedule,
    )
    if policy.mlp_extractor.value_net is not policy.mlp_extractor.policy_net:
        raise RuntimeError("SharedTrunkPolicy trunk sharing failed")
    return policy


def load_into_sb3(policy: nn.Module, sd: dict[str, torch.Tensor]) -> None:
    """Map reference tensors onto an SB3 ActorCriticPolicy.

    Trunk (shared via SharedTrunkPolicy) lives in
    ``policy.mlp_extractor.policy_net`` (identical module instance also
    registered as ``mlp_extractor.value_net``); heads are ``policy.action_net``
    and ``policy.value_net``.
    """
    extractor = policy.mlp_extractor.policy_net  # type: ignore[union-attr]
    mapping = {
        "extract_layer.0.weight": (extractor, 0, "weight"),
        "extract_layer.0.bias": (extractor, 0, "bias"),
        "extract_layer.2.weight": (extractor, 2, "weight"),
        "extract_layer.2.bias": (extractor, 2, "bias"),
        "actor.weight": (policy.action_net, None, "weight"),   # type: ignore[attr-defined]
        "actor.bias": (policy.action_net, None, "bias"),
        "critic.weight": (policy.value_net, None, "weight"),   # type: ignore[attr-defined]
        "critic.bias": (policy.value_net, None, "bias"),
    }
    for ref_key, (module, index, attr) in mapping.items():
        layer = module[index] if index is not None else module
        with torch.no_grad():
            getattr(layer, attr).copy_(sd[ref_key])
    read_back = {
        "extract_layer.0.weight": extractor[0].weight,
        "extract_layer.0.bias": extractor[0].bias,
        "extract_layer.2.weight": extractor[2].weight,
        "extract_layer.2.bias": extractor[2].bias,
        "actor.weight": policy.action_net.weight,   # type: ignore[attr-defined]
        "actor.bias": policy.action_net.bias,
        "critic.weight": policy.value_net.weight,   # type: ignore[attr-defined]
        "critic.bias": policy.value_net.bias,
    }
    _assert_equal(read_back, sd, "SB3 policy")


# ---------------------------------------------------------------------------
# TorchRL adapter
# ---------------------------------------------------------------------------

class TorchRLSharedTrunk(nn.Module):
    """Shared 64-64 Tanh trunk + actor/critic heads (zeroRL topology).

    ``actor_path`` / ``critic_path`` are ``nn.Sequential`` objects that hold
    *the same* trunk instance, so parameters are shared and de-duplicated by
    ``parameters()`` — matching zeroRL's single-shared-trunk agent exactly.
    """

    def __init__(self, sd: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(HIDDEN, ACT_DIM)
        self.critic_head = nn.Linear(HIDDEN, 1)
        sd = sd if sd is not None else build_reference_state_dict()
        with torch.no_grad():
            self.trunk[0].weight.copy_(sd["extract_layer.0.weight"])
            self.trunk[0].bias.copy_(sd["extract_layer.0.bias"])
            self.trunk[2].weight.copy_(sd["extract_layer.2.weight"])
            self.trunk[2].bias.copy_(sd["extract_layer.2.bias"])
            self.actor_head.weight.copy_(sd["actor.weight"])
            self.actor_head.bias.copy_(sd["actor.bias"])
            self.critic_head.weight.copy_(sd["critic.weight"])
            self.critic_head.bias.copy_(sd["critic.bias"])
        # expose paths that share the trunk instance (no parameter copies)
        self.actor_path = nn.Sequential(self.trunk, self.actor_head)
        self.critic_path = nn.Sequential(self.trunk, self.critic_head)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic argmax action, shape (B,) int64."""
        return self.actor_path(obs).argmax(-1).long()

    def ortho_init_heads(self) -> None:  # pragma: no cover - kept for clarity
        raise NotImplementedError("weights are loaded from the reference, not reinited")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _assert_equal(actual: dict[str, torch.Tensor],
                  expected: dict[str, torch.Tensor],
                  label: str) -> None:
    for key, ref in expected.items():
        if key not in actual:
            raise AssertionError(f"{label}: key {key!r} missing after load")
        if not torch.equal(actual[key].cpu(), ref.cpu()):
            diff = (actual[key].cpu() - ref.cpu()).abs().max().item()
            raise AssertionError(f"{label}: tensor {key!r} differs (max |Δ|={diff})")


def assert_deterministic_parity(seed: int = REFERENCE_SEED) -> dict[str, float]:
    """Cross-adapter sanity: all three loads produce identical argmax actions
    on a fixed batch of synthetic observations. Returns the action vector as
    a float fingerprint for logging."""
    sd = build_reference_state_dict(seed)
    obs = torch.as_tensor(
        np.stack([np.linspace(-1.0, 1.0, OBS_DIM) * k for k in (1.0, -0.5, 0.25, 2.0)]),
        dtype=torch.float32,
    )

    z_agent = build_deterministic_zerorl_agent(sd)
    z_action = z_agent.get_action(obs)["action"]

    trunk = TorchRLSharedTrunk(sd)
    t_action = trunk.act(obs)

    if not torch.equal(z_action, t_action):
        raise AssertionError(
            f"zeroRL vs TorchRL argmax mismatch: {z_action.tolist()} vs {t_action.tolist()}"
        )
    return {"fingerprint": float(z_action.double().sum())}


def assert_deterministic_parity_with_sb3(policy: nn.Module,
                                         seed: int = REFERENCE_SEED) -> None:
    """Extend the parity check to a loaded SB3 policy (needs sb3 installed).

    Uses the official deterministic ``predict`` path — the exact call the
    benchmark's SB3 collection loop makes.
    """
    sd = build_reference_state_dict(seed)
    obs = torch.as_tensor(
        np.stack([np.linspace(-1.0, 1.0, OBS_DIM) * k for k in (1.0, -0.5, 0.25, 2.0)]),
        dtype=torch.float32,
    )
    z_agent = build_deterministic_zerorl_agent(sd)
    z_action = z_agent.get_action(obs)["action"]
    sb3_action, _ = policy.predict(obs.numpy(), deterministic=True)
    if not torch.equal(z_action, torch.as_tensor(sb3_action).long().cpu()):
        raise AssertionError(
            f"zeroRL vs SB3 argmax mismatch: {z_action.tolist()} vs {list(sb3_action)}"
        )
