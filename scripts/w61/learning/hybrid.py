"""Common bounded two-chart policy, analytic mixed KL and independently keyed random streams."""
import hashlib
from bisect import bisect_right
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from s1_w59_model_fields import NS

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/terrain_memory/w61/learning/hybrid.json"
SPEC = json.loads(CONFIG.read_text())
RESULT = ROOT / "results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/learning"
CACHE = ROOT / "work/w61/learning" / SPEC["identity"]
OBS = NS + 5
DIMS = 4
ACTIVE = torch.tensor([[1., 1., 1., 1.], [0., 0., 0., 1.]])
SIGMA = SPEC["normal_std"]


def original_key(seed, *identity):
    material = json.dumps([int(seed), *identity], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "little")


MASK32 = (1<<32)-1
MULTIPLIER = 0x9E3779B1
SHIFT = 0x85EBCA6B


@lru_cache(maxsize=1)
def reserved_ordinals():
    # Keep all original primary/restart-1 draws. Invert an odd affine permutation to
    # count the unused 32-bit slots in a fixed, reproducible order.
    occupied = {original_key(seed,"train","explore",case,version,episode,n) & MASK32
        for seed in SPEC["seeds"] for case in SPEC["scenarios"] for version in range(32)
        for episode in (0,1) for n in range(128)}
    if len(occupied)!=49152:
        raise ValueError("Registered primary and first restart no longer have 49152 distinct effective keys")
    inverse=pow(MULTIPLIER,-1,1<<32)
    return tuple(sorted(((k-SHIFT)*inverse)&MASK32 for k in occupied))


def key(seed, *identity):
    old=original_key(seed,*identity)
    if len(identity)!=6 or identity[:2]!=("train","explore") or identity[4]<2:
        return old
    _,_,case,version,episode,number=identity
    if not (0<=version<32 and 2<=episode<=130 and 0<=number<128):
        raise ValueError("Training request exceeds its preregistered distinct stream domain")
    rank=((((SPEC["seeds"].index(seed)*2+SPEC["scenarios"].index(case))*32+version)*129+episode-2)*128+number)
    reserve=reserved_ordinals()
    slot=rank
    while True:
        advanced=rank+bisect_right(reserve,slot)
        if advanced==slot:
            break
        slot=advanced
    return (old & ~MASK32) | ((MULTIPLIER*slot+SHIFT)&MASK32)


def stream(seed, *identity):
    return torch.Generator(device="cpu").manual_seed(key(seed, *identity))


def weight_digest(module):
    h = hashlib.sha256()
    for name, param in module.state_dict().items():
        h.update(name.encode())
        h.update(param.contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def network(out):
    return nn.Sequential(nn.Linear(OBS, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, out))


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = network(2 + 2 * DIMS)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[:2].copy_(torch.tensor(SPEC["mode_prior"]).log())

    def forward(self, obs):
        raw = self.net(obs.asinh())
        logits = raw[..., :2].masked_fill(obs[..., -2:] < .5, -1.e9)
        return logits.log_softmax(-1), raw[..., 2:].reshape(*raw.shape[:-1], 2, DIMS)


class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = network(1)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        return self.net(obs.asinh()).squeeze(-1)


def new_heads(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(key(seed, "initial_weights"))
        return Policy(), Value()


def joint_logp(logp, means, modes, raw):
    indices = torch.arange(len(modes), device=modes.device)
    subset = ACTIVE.to(means)[modes]
    normal = -.5 * ((raw - means[indices, modes]) / SIGMA).square() - math.log(SIGMA * math.sqrt(2*math.pi))
    return logp[indices, modes] + (subset * normal).sum(-1)


def mixed_kl(oldlog, oldmean, logp, mean):
    gaussian = .5 * (((mean-oldmean) / SIGMA).square() * ACTIVE.to(mean)).sum(-1)
    return (oldlog.exp() * (oldlog - logp + gaussian)).sum(-1).mean()


def action_map(mode, raw):
    u = np.tanh(np.asarray(raw))
    if mode == 1:
        return np.array([0., 0., 0., SPEC["hold_center_s"] + u[3] * SPEC["hold_radius_s"]])
    return np.asarray(SPEC["transfer_center"]) + np.asarray(SPEC["transfer_radius"]) * u


def rng_receipt():
    from s1_w59_learner_critic import atomic_json
    def sample(generator):
        draws = torch.randn(32, generator=generator)
        return dict(first=draws[:4].tolist(), sha256=hashlib.sha256(draws.numpy().tobytes()).hexdigest(),
                    state_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest())
    old = {str(s):sample(torch.Generator().manual_seed(s*2**32+11)) for s in SPEC["seeds"]}
    new = {str(s):{case:sample(stream(s, "train", "explore", case, 0, 0, 11))
                   for case in SPEC["scenarios"]} for s in SPEC["seeds"]}
    result = dict(torch=torch.__version__, engine="torch.Generator CPU", old=old, new=new,
                  old_independent_fingerprints=len({a["sha256"] for a in old.values()}),
                  new_independent_fingerprints=len({a["sha256"] for cases in new.values() for a in cases.values()}),
                  recovery="per-request blake2b64(seed,role,case,version,episode,event index); key and pre-draw state digest recorded, checkpoint stores rollout cursor",
                  nominal_same_seed_pairing="common initial weights, scene, RNG keys; separate physics and per-head event IDs")
    atomic_json(RESULT/"rng-d57-v1.json",result)
    return result


if __name__ == "__main__":
    print(json.dumps(rng_receipt(), separators=(",", ":")))
