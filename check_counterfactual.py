"""
check_counterfactual.py
=======================
Logic checks for the stronger CMC anchors (KL anchor, counterfactual selection,
counterfactual re-prioritization). Needs only torch + numpy; uses the fake
environment from sanity_check.py, so put both files next to run_experiment.py.

    python check_counterfactual.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from cmc_rl.replay import ReplayBuffer, CMCMemory
from cmc_rl.cmc import CounterfactualMemoryConsolidator, gaussian_kl
from cmc_rl.tasks import TASK_SEQUENCE
import run_experiment as rx
from sanity_check import OBS, ACT, DEV, RESULTS, check, fake_factory, make_agent


def random_replay(n=2000):
    rb = ReplayBuffer(n, OBS, ACT)
    for _ in range(n):
        rb.add(np.random.randn(OBS), np.random.uniform(-1, 1, ACT), np.random.randn(),
               np.random.randn(OBS), 0.0)
    return rb


def perturb(module, scale=0.5):
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn_like(p))


@check("gaussian_kl: zero for identical Gaussians, matches torch.distributions")
def t_kl_formula():
    mu0, ls0 = torch.randn(64, ACT), torch.randn(64, ACT) * 0.3
    mu1, ls1 = torch.randn(64, ACT), torch.randn(64, ACT) * 0.3
    assert gaussian_kl(mu0, ls0, mu0, ls0).abs().max() < 1e-6
    ref = torch.distributions.kl_divergence(torch.distributions.Normal(mu0, ls0.exp()),
                                            torch.distributions.Normal(mu1, ls1.exp()))
    assert torch.allclose(gaussian_kl(mu0, ls0, mu1, ls1), ref, atol=1e-5)


@check("Consolidation stores the old policy's mean and log-std")
def t_stores_dist():
    agent = make_agent()
    cmc = CounterfactualMemoryConsolidator(CMCMemory(200, OBS, ACT))
    cmc.consolidate_task(agent, random_replay(), 0, max_candidates=500)
    m = cmc.memory
    assert m.has_dist
    d = agent.actor.distribution(torch.as_tensor(m.obs[:16]))
    assert np.allclose(m.old_mean[:16], d.loc.detach().numpy(), atol=1e-5)
    assert np.allclose(m.old_log_std[:16], d.scale.log().detach().numpy(), atol=1e-5)


@check("KL anchor: zero at the stored policy, has gradient, also restores the std (MSE cannot)")
def t_kl_anchor():
    agent = make_agent()
    cmc = CounterfactualMemoryConsolidator(CMCMemory(256, OBS, ACT), actor_anchor="kl")
    cmc.consolidate_task(agent, random_replay(), 0, max_candidates=1000)
    assert cmc.actor_loss(agent, 256, DEV).item() < 1e-5
    # Same means, wider stored std: MSE sees nothing, KL does.
    cmc.memory.old_log_std = cmc.memory.old_log_std + 0.7
    mse = CounterfactualMemoryConsolidator(cmc.memory, actor_anchor="mse")
    assert mse.actor_loss(agent, 256, DEV).item() < 1e-8
    l0 = cmc.actor_loss(agent, 256, DEV)
    assert l0.requires_grad and l0.item() > 0.1
    gap = lambda: float((cmc.memory.old_log_std - agent.actor.distribution(
        torch.as_tensor(cmc.memory.obs)).scale.log().detach().numpy()).mean())
    g0 = gap()
    for _ in range(200):
        l = cmc.actor_loss(agent, 256, DEV)
        agent.actor_opt.zero_grad(); l.backward(); agent.actor_opt.step()
    assert gap() < 0.3 * g0, (g0, gap())


@check("KL anchor refuses a memory without stored distributions")
def t_kl_needs_dist():
    mem = CMCMemory(100, OBS, ACT)
    mem.add_task(0, np.random.randn(100, OBS), np.zeros((100, ACT)), np.zeros(100), np.ones(100))
    cmc = CounterfactualMemoryConsolidator(mem, actor_anchor="kl")
    try:
        cmc.actor_loss(make_agent(), 32, DEV)
        raise AssertionError("KL anchor ran without old_mean/old_log_std")
    except RuntimeError:
        pass


@check("Counterfactual selection: needs begin_task; keeps the largest value gains over the pre-task actor")
def t_cf_selection():
    agent = make_agent()
    cmc = CounterfactualMemoryConsolidator(CMCMemory(100, OBS, ACT), selection="counterfactual")
    rb = random_replay()
    try:
        cmc.consolidate_task(agent, rb, 0, max_candidates=500)
        raise AssertionError("counterfactual selection ran without a pre-task actor")
    except RuntimeError:
        pass
    cmc.begin_task(agent, 0)
    pre = cmc._pre_task_actor
    perturb(agent.actor)                               # "learning" the task
    assert not torch.allclose(pre.deterministic(torch.zeros(1, OBS)),
                              agent.actor.deterministic(torch.zeros(1, OBS)))
    info = cmc.consolidate_task(agent, rb, 0, max_candidates=500)
    m = cmc.memory
    s = torch.as_tensor(m.obs)
    with torch.no_grad():
        q = lambda a: torch.minimum(agent.q1(s, a), agent.q2(s, a)).squeeze(-1)
        gain = (q(agent.actor.deterministic(s)) - q(pre.deterministic(s))).numpy()
    assert np.allclose(m.priority, gain, atol=1e-4), "priority is not the counterfactual gain"
    assert np.all(np.diff(m.priority) <= 1e-6), "memory not sorted by gain"
    assert info["kept"] == 100 and "cf_gain_mean" in info
    # The pre-task actor is frozen: it did not move with the trained actor.
    assert all(not p.requires_grad for p in pre.parameters())


@check("Counterfactual refresh: zero regret -> uniform; drift -> high-regret states sampled more; task balance kept")
def t_cf_refresh():
    agent = make_agent()
    mem = CMCMemory(400, OBS, ACT)
    cmc = CounterfactualMemoryConsolidator(mem, cf_refresh_every=1, cf_eps=0.05)
    for tid in (0, 1):
        cmc.begin_task(agent, tid)
        cmc.consolidate_task(agent, random_replay(1000), tid, max_candidates=600)
    assert set(cmc.snapshots) == {0, 1}
    r = cmc.counterfactual_regret(agent, DEV)
    assert np.abs(r).max() < 1e-4, "regret should be 0 when the policy has not changed"
    cmc.refresh_priorities(agent, DEV)
    assert np.allclose(mem._p, 1.0 / mem.size), "zero regret must give uniform sampling"

    perturb(agent.actor)
    stats = cmc.on_update(agent, DEV, stage="T3", step=1)
    assert stats and set(stats) == {0, 1} and cmc.refresh_log[-1]["step"] == 1
    r = cmc.counterfactual_regret(agent, DEV)
    for tid in (0, 1):
        assert abs(mem._p[mem.task_id == tid].sum() - 0.5) < 1e-9, "task balance broken"
    idx = np.random.choice(mem.size, 200_000, p=mem._p)
    counts = np.bincount(idx, minlength=mem.size)
    hi, lo = r > np.percentile(r, 80), r <= np.percentile(r, 20)
    assert counts[hi].mean() > 2 * counts[lo].mean(), "high-regret states not prioritized"
    # adding a new task resets to uniform until the next refresh
    mem.add_task(2, np.random.randn(50, OBS), np.zeros((50, ACT)), np.zeros(50), np.ones(50))
    assert mem._p is None


@check("Training loop: KL anchor + counterfactual selection + refresh run end to end")
def t_train_cf():
    agent = make_agent()
    args = rx.parse_args(["--steps-per-task", "1200", "--warmup", "200", "--batch-size", "32",
                          "--eval-every", "0", "--log-every", "100000", "--cmc-candidates", "500",
                          "--actor-anchor", "kl", "--memory-selection", "counterfactual",
                          "--cf-refresh-every", "50"])
    cmc = CounterfactualMemoryConsolidator(
        CMCMemory(300, OBS, ACT), selection=args.memory_selection,
        actor_anchor=args.actor_anchor, cf_refresh_every=args.cf_refresh_every)
    for k in (0, 1):
        cmc.begin_task(agent, k)
        info = rx.train_one_task(agent, TASK_SEQUENCE[k], TASK_SEQUENCE, k, args,
                                 ReplayBuffer(2000, OBS, ACT), cmc, True, DEV, [],
                                 env_fn=fake_factory())
        assert info["consolidation"] and "cf_gain_mean" in info["consolidation"]
    assert cmc.memory.counts_per_task() == {0: 150, 1: 150} and cmc.memory.has_dist
    assert len(cmc.refresh_log) > 0, "no counterfactual refresh during T2"
    assert all(e["stage"] == "T2" for e in cmc.refresh_log)
    json.dumps(cmc.refresh_log)                        # must be JSON-serializable


@check("Output guard: a different CMC variant cannot overwrite an existing CMC run")
def t_guard():
    d = Path(tempfile.mkdtemp())
    (d / "config.json").write_text(json.dumps({"args": {}}))      # old run, default variant
    rx.guard_existing_run(d, rx.parse_args([]))                    # same variant: fine
    try:
        rx.guard_existing_run(d, rx.parse_args(["--actor-anchor", "kl"]))
        raise AssertionError("variant mismatch was not caught")
    except SystemExit:
        pass
    rx.guard_existing_run(d, rx.parse_args(["--no-cmc"]))          # baseline unaffected


def main():
    for t in [t_kl_formula, t_stores_dist, t_kl_anchor, t_kl_needs_dist,
              t_cf_selection, t_cf_refresh, t_train_cf, t_guard]:
        t()
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} counterfactual-anchor checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
