"""CMC-inspired memory consolidation for SAC.

Changes vs. the original:
  * FIX 1: the actor anchor uses agent.actor.mean_action() (differentiable).
    The old code used actor.deterministic(), which is @torch.no_grad(), so the
    actor regularizer produced zero gradient and CMC never constrained the actor.
  * FIX 4: candidate states come from the most recent `recent_frac` of the task's
    replay buffer (skips warmup / early-training states).
  * FIX 4: the priority score is fixed and made scale-free:
       - policy disagreement  |mu(s) - a~pi(s)|^2
       - Q gap                |Q(s, mu(s)) - Q(s, a~)|
       - real TD error        |r + g(1-term)(minQt(s', a') - alpha log pi(a'|s')) - minQ(s, a)|
         (the old proxy bootstrapped from s instead of s', so it was not a TD error)
     Each term is rank-normalized to [0, 1] within the task, then averaged, so
     no single term (or task reward scale) dominates.
  * selection="uniform" gives the random-memory control (ablation).
  * Separate actor_loss() / critic_loss() so each update samples memory once.

Stronger anchors (all opt-in; the defaults reproduce the runs above exactly):
  * actor_anchor="kl": anchor the FULL action distribution, not only its mean.
    At consolidation the pre-tanh Gaussian N(mu_old, sigma_old) is stored, and the
    loss is KL(pi_old(.|s) || pi_new(.|s)), summed over action dimensions.
    tanh-squashing plus the affine rescale is a bijection, and KL divergence is
    invariant under bijections, so the pre-tanh KL is EXACTLY the KL between the
    squashed SAC policies (no approximation, no sampling).
  * selection="counterfactual": a frozen copy of the actor is taken at the START
    of each task (begin_task). At consolidation every candidate state is scored by
        gain_k(s) = minQ_k(s, pi_k(s)) - minQ_k(s, pi_pre(s)),
    i.e. how much task-k value would be lost if the agent acted as it did before
    learning task k. The heuristic score asks "is this state uncertain?"; this one
    asks "does what was learned here matter?", a true counterfactual comparison
    between two policies at the same state, judged by the same critic.
  * cf_refresh_every=N: at consolidation, frozen copies of task k's critics are
    kept. Every N actor updates on later tasks, each memory state of task k gets
        regret_k(s) = minQ_k^frozen(s, a_old) - minQ_k^frozen(s, pi_current(s)),
    the value on task k lost by acting with the CURRENT policy instead of the old
    one. The memory is then sampled in proportion to (normalized regret + eps)^beta
    within each task (equal per-task mass is kept), so the anchor concentrates on
    states that are actually being forgotten. The regret trace is logged
    (refresh_log) and is a direct, per-task measure of counterfactual forgetting.
    Caveat: Q_k^frozen(s, pi_current(s)) is an off-distribution query of an old
    critic, so it can over- or under-estimate; regret is used to RANK states
    within a task, not as an absolute measure of lost return.
"""

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F


ANCHORS = ("mse", "kl")
SELECTIONS = ("priority", "uniform", "counterfactual")


def _rank01(x):
    """Rank-normalize a 1-D tensor to [0, 1] (ties broken arbitrarily)."""
    n = x.numel()
    if n <= 1:
        return torch.zeros_like(x)
    ranks = torch.empty_like(x)
    ranks[torch.argsort(x)] = torch.arange(n, dtype=x.dtype, device=x.device)
    return ranks / (n - 1)


def _frozen(module):
    """Gradient-free deep copy of a network (a snapshot that never trains)."""
    m = copy.deepcopy(module)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _min_q(q1, q2, obs, act):
    return torch.minimum(q1(obs, act), q2(obs, act)).squeeze(-1)


def gaussian_kl(mu_old, log_std_old, mu_new, log_std_new):
    """KL(N(mu_old, s_old) || N(mu_new, s_new)) per dimension (diagonal Gaussians)."""
    var_old = torch.exp(2.0 * log_std_old)
    var_new = torch.exp(2.0 * log_std_new)
    return (log_std_new - log_std_old
            + (var_old + (mu_old - mu_new) ** 2) / (2.0 * var_new) - 0.5)


class CounterfactualMemoryConsolidator:
    def __init__(self, memory, actor_lambda=1.0, critic_lambda=0.25,
                 selection="priority", recent_frac=0.3, gamma=0.99,
                 actor_anchor="mse", kl_min_log_std=-5.0,
                 cf_refresh_every=0, cf_beta=1.0, cf_eps=0.1, cf_batch=4096):
        assert selection in SELECTIONS, selection
        assert actor_anchor in ANCHORS, actor_anchor
        self.memory = memory
        self.actor_lambda = actor_lambda
        self.critic_lambda = critic_lambda
        self.selection = selection
        self.recent_frac = recent_frac
        self.gamma = gamma
        self.actor_anchor = actor_anchor
        self.kl_min_log_std = kl_min_log_std
        self.cf_refresh_every = int(cf_refresh_every)
        self.cf_beta = cf_beta
        self.cf_eps = cf_eps
        self.cf_batch = cf_batch

        self._pre_task_actor = None   # frozen actor from the start of the current task
        self.snapshots = {}           # task_id -> frozen {"actor", "q1", "q2"} at consolidation
        self._n_updates = 0
        self.last_refresh = None      # most recent per-task regret summary
        self.refresh_log = []         # every refresh: {"stage", "step", task_id: stats}

    # ------------------------------------------------------------ task hooks
    def begin_task(self, agent, task_id=None):
        """Call before training each task: freezes the pre-task actor."""
        self._pre_task_actor = _frozen(agent.actor)
        self._n_updates = 0

    def on_update(self, agent, device, stage=None, step=None):
        """Call after every actor update. Refreshes counterfactual priorities
        every `cf_refresh_every` updates; returns the stats when it did."""
        if self.cf_refresh_every <= 0 or self.memory.size == 0:
            return None
        self._n_updates += 1
        if self._n_updates % self.cf_refresh_every:
            return None
        stats = self.refresh_priorities(agent, device)
        self.refresh_log.append({"stage": stage, "step": step,
                                 **{str(k): v for k, v in stats.items()}})
        return stats

    # ----------------------------------------------------------- consolidation
    @torch.no_grad()
    def consolidate_task(self, agent, replay, task_id, max_candidates=5000,
                         device="cpu"):
        pool = replay.recent_indices(self.recent_frac)
        if len(pool) == 0 or self.memory.capacity <= 0:
            return {"candidates": 0, "kept": 0,
                    "per_task": self.memory.counts_per_task()}

        n = min(max_candidates, len(pool))
        idx = np.random.choice(pool, n, replace=False)

        obs_np = replay.obs[idx]
        nobs_np = replay.next_obs[idx]
        rew_np = replay.rewards[idx]
        finite = (np.isfinite(obs_np).all(1) & np.isfinite(nobs_np).all(1)
                  & np.isfinite(rew_np))
        idx = idx[finite]
        if len(idx) == 0:
            return {"candidates": 0, "kept": 0,
                    "per_task": self.memory.counts_per_task()}

        t = lambda a: torch.as_tensor(a, device=device)
        obs = t(replay.obs[idx])
        act = t(replay.actions[idx])
        rew = t(replay.rewards[idx])
        nobs = t(replay.next_obs[idx])
        term = t(replay.terminals[idx])

        old_action = agent.actor.deterministic(obs)
        old_q = _min_q(agent.q1, agent.q2, obs, old_action)
        dist = agent.actor.distribution(obs)
        old_mean, old_log_std = dist.loc, dist.scale.log()
        extra = {}

        if self.selection == "uniform":
            priority = torch.rand(len(idx), device=device)
        elif self.selection == "counterfactual":
            if self._pre_task_actor is None:
                raise RuntimeError("selection='counterfactual' needs begin_task(agent) "
                                   "to be called before the task is trained")
            pre_action = self._pre_task_actor.deterministic(obs)
            gain = old_q - _min_q(agent.q1, agent.q2, obs, pre_action)
            priority = gain                      # only compared within this task
            extra = {"cf_gain_mean": float(gain.mean()),
                     "cf_gain_frac_positive": float((gain > 0).float().mean())}
        else:
            stoch_a, _ = agent.actor.sample(obs)
            q_stoch = _min_q(agent.q1, agent.q2, obs, stoch_a)
            disagreement = ((old_action - stoch_a) ** 2).mean(-1)
            q_gap = (old_q - q_stoch).abs()

            na, nlogp = agent.actor.sample(nobs)
            tq = _min_q(agent.q1_target, agent.q2_target, nobs, na)
            target = rew + self.gamma * (1.0 - term) * (tq - agent.alpha * nlogp.squeeze(-1))
            q_sa = _min_q(agent.q1, agent.q2, obs, act)
            td = (target - q_sa).abs()

            priority = (_rank01(disagreement) + _rank01(q_gap) + _rank01(td)) / 3.0

        kept = self.memory.add_task(
            task_id,
            obs.cpu().numpy(), old_action.cpu().numpy(),
            old_q.cpu().numpy(), priority.cpu().numpy(),
            old_mean=old_mean.cpu().numpy(), old_log_std=old_log_std.cpu().numpy(),
        )
        if self.cf_refresh_every > 0:
            self.snapshots[int(task_id)] = {"actor": _frozen(agent.actor),
                                            "q1": _frozen(agent.q1),
                                            "q2": _frozen(agent.q2)}
        return {"candidates": int(len(idx)), "kept": int(kept),
                "per_task": self.memory.counts_per_task(), **extra}

    # ------------------------------------------------- counterfactual refresh
    @torch.no_grad()
    def counterfactual_regret(self, agent, device):
        """regret(s) = minQ_k^frozen(s, a_old) - minQ_k^frozen(s, pi_now(s)) for every
        memory state, with k the task that stored it. NaN where no snapshot exists."""
        m = self.memory
        regret = np.full(m.size, np.nan, np.float32)
        for tid in np.unique(m.task_id):
            snap = self.snapshots.get(int(tid))
            if snap is None:
                continue
            sel = np.nonzero(m.task_id == tid)[0]
            for chunk in np.array_split(sel, max(1, math.ceil(len(sel) / self.cf_batch))):
                s = torch.as_tensor(m.obs[chunk], device=device)
                a_old = torch.as_tensor(m.old_action[chunk], device=device)
                a_now = agent.actor.deterministic(s)
                r = (_min_q(snap["q1"], snap["q2"], s, a_old)
                     - _min_q(snap["q1"], snap["q2"], s, a_now))
                regret[chunk] = r.cpu().numpy()
        return regret

    def refresh_priorities(self, agent, device):
        m = self.memory
        regret = self.counterfactual_regret(agent, device)
        weights = np.ones(m.size, np.float64)
        stats = {}
        for tid in np.unique(m.task_id):
            sel = m.task_id == tid
            r = regret[sel]
            if np.isnan(r).all():
                continue
            pos = np.maximum(np.nan_to_num(r, nan=0.0), 0.0)
            scale = pos.mean()
            w = (pos / scale if scale > 0 else np.zeros_like(pos)) + self.cf_eps
            weights[sel] = w ** self.cf_beta
            stats[int(tid)] = {"regret_mean": float(np.nanmean(r)),
                               "regret_p90": float(np.nanpercentile(r, 90)),
                               "frac_positive": float((r > 0).mean())}
        m.set_sampling_weights(weights)
        self.last_refresh = stats
        return stats

    # ------------------------------------------------------------------ losses
    def actor_loss(self, agent, batch_size, device):
        """Anchor the CURRENT (differentiable) policy to the stored one.

        mse: squared error between tanh means (the original CMC anchor).
        kl : KL(pi_old || pi_new) of the pre-tanh Gaussians = exact KL of the
             squashed policies. log-stds are floored at kl_min_log_std so a
             near-deterministic stored policy cannot make the loss explode.
        """
        if self.actor_anchor == "mse":
            batch = self.memory.sample(batch_size, device)
            if batch is None:
                return torch.zeros((), device=device)
            mem_obs, old_action, _, _ = batch
            new_action = agent.actor.mean_action(mem_obs)
            return F.mse_loss(new_action, old_action)

        batch = self.memory.sample(batch_size, device, with_dist=True)
        if batch is None:
            return torch.zeros((), device=device)
        mem_obs, _, _, _, old_mean, old_log_std = batch
        dist = agent.actor.distribution(mem_obs)
        lo = self.kl_min_log_std
        kl = gaussian_kl(old_mean, old_log_std.clamp(min=lo),
                         dist.loc, dist.scale.log().clamp(min=lo))
        return kl.sum(-1).mean()

    def critic_loss(self, agent, batch_size, device):
        """Anchor both critics at (s_mem, a_old) to the stored Q value."""
        batch = self.memory.sample(batch_size, device)
        if batch is None:
            return torch.zeros((), device=device)
        mem_obs, old_action, old_q, _ = batch
        q1 = agent.q1(mem_obs, old_action).squeeze(-1)
        q2 = agent.q2(mem_obs, old_action).squeeze(-1)
        return 0.5 * (F.mse_loss(q1, old_q) + F.mse_loss(q2, old_q))

    def loss(self, agent, obs):
        """Backward-compatible API: returns (actor_loss, critic_loss)."""
        device = obs.device
        return (self.actor_loss(agent, len(obs), device),
                self.critic_loss(agent, len(obs), device))
