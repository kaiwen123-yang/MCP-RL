"""The W57 event-duration PPO adapter, checkpoints and two-collector policy barrier."""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tianshou.data import Batch

from response_ppo.ppo_backend import PPOBackend, PPOConfig


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT/'experiments/terrain_memory/s1-w57-timing.json'
SPEC = json.loads(SPEC_PATH.read_text())
SPEC_HASH = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
STEPS = SPEC['ppo']['rollout_shape'][0]


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def jsonline(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, default=native, allow_nan=False, separators=(',',':'))+'\n')


def atomic_json(path, row):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(row, default=native, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def checkpoint_path(run, version):
    return run/f'policy-{version:03d}.pt'


def batch_path(run, version, case):
    return run/f'batch-{version:03d}-{case}.jsonl'


def backend(device):
    p = SPEC['ppo']
    return PPOBackend(PPOConfig(obs_dim=SPEC['normalization']['dimension'],
        action_bound=1., hidden=tuple(p['hidden']), learning_rate=p['learning_rate'],
        initial_std=p['initial_std'], gamma=p['gamma'], gae_lambda=p['gae_lambda'],
        clip_epsilon=p['clip'], value_coef=p['value_coef'], entropy_coef=p['entropy_coef'],
        max_grad_norm=p['max_grad_norm'], epochs=p['epochs'], minibatch_size=p['minibatch'],
        device=device))


def restore(path, device, optimizer=False):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['spec_sha256'] != SPEC_HASH:
        raise ValueError('Checkpoint belongs to a different event, schema or learning contract.')
    # Constructing a second model may not consume the old actor or physics RNG.
    with torch.random.fork_rng(devices=[]):
        model = backend(device)
    model.actor.load_state_dict(payload['actor'])
    model.critic.load_state_dict(payload['critic'])
    model.update_count = payload['version']
    if optimizer:
        model.optimizer.load_state_dict(payload['optimizer'])
        torch.set_rng_state(payload['cpu_rng'])
        np.random.set_state(payload['numpy_rng'])
        if device.startswith('cuda'):
            if payload['cuda_rng'] is None:
                torch.cuda.manual_seed(SPEC['learning_seed'])
            else:
                torch.cuda.set_rng_state(payload['cuda_rng'])
    return model, payload


def save_policy(path, model, version, head, identity, metrics=None):
    payload = dict(schema=SPEC['identity'], spec_sha256=SPEC_HASH, spec=SPEC,
        head=head, identity=identity, version=version, samples=version*2*STEPS,
        actor=model.actor.state_dict(), critic=model.critic.state_dict(),
        optimizer=model.optimizer.state_dict(), cpu_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(), cuda_rng=(torch.cuda.get_rng_state()
        if model.config.device.startswith('cuda') else None), collector_next_position={'A':STEPS*version,
        'B':STEPS*version}, learning_seed=SPEC['learning_seed'], metrics=metrics)
    tmp = path.with_suffix('.pt.tmp')
    torch.save(payload,tmp)
    os.replace(tmp,path)


def common_initialization(path):
    torch.manual_seed(SPEC['learning_seed'])
    np.random.seed(SPEC['learning_seed'])
    model = backend('cpu')
    save_policy(path, model, 0, 'common', 'same initialization for full and blind')


class EventPolicy:
    """Only completed 128-row segments advance a shared version; physics waits in place."""
    def __init__(self, config):
        self.mode = config['mode']
        self.head = config['head']
        self.case = config.get('case')
        self.position = 0
        self.finished = self.mode == 'evaluate'
        if self.mode == 'collect':
            self.run = Path(config['run_dir'])
            manifest = json.loads((self.run/'run.json').read_text())
            if self.head != manifest['head'] or self.case not in SPEC['cases']:
                raise ValueError('Collector case/head disagree with the frozen learning identity.')
            version = json.loads((self.run/'published.json').read_text())['version']
            path = checkpoint_path(self.run, version)
            self.position = len(read_rows(batch_path(self.run, version, self.case)))
        elif self.mode == 'evaluate':
            path = Path(config['checkpoint'])
        else:
            raise ValueError(f'Unknown timing mode {self.mode}')
        self.install(path)

    def install(self, path):
        self.model, payload = restore(path,'cpu')
        if payload['head'] != self.head:
            raise ValueError('The selected head and installed checkpoint disagree.')
        self.version = int(payload['version'])

    @torch.no_grad()
    def value(self, obs):
        return float(self.model.value(torch.from_numpy(obs)[None]).item())

    @torch.no_grad()
    def mean(self, obs):
        (mean,_), _ = self.model.actor(torch.from_numpy(obs)[None])
        return float(mean.tanh().item())

    def wait_at_barrier(self):
        while True:
            publication = json.loads((self.run/'published.json').read_text())
            if publication['version'] > self.version:
                self.install(checkpoint_path(self.run,publication['version']))
                self.position = 0
                self.finished = publication['finished']
                return
            # A node decision can stop before another update, e.g. a diagnosed batch error.
            if publication.get('error'):
                raise RuntimeError(publication['error'])
            if publication['finished']:
                self.finished = True
                return
            time.sleep(1.)

    @torch.no_grad()
    def action(self, obs):
        if not self.finished and self.position == STEPS:
            self.wait_at_barrier()
        generator = None
        seed = None
        serial = None
        if not self.finished:
            serial = self.version*STEPS+self.position
            seed = SPEC['learning_seed']*2**32+2*serial+SPEC['cases'].index(self.case)
            generator = torch.Generator(device='cpu').manual_seed(seed)
        input_tensor = torch.from_numpy(obs)[None]
        sampled = self.model.act(input_tensor, deterministic=self.finished, generator=generator)
        return dict(version=self.version, sampling='on_policy' if not self.finished else 'frozen_mean',
            sample_serial=serial, rng_seed=seed, raw_action=float(sampled.raw_action.item()),
            action=float(sampled.action.item()), logp_raw=float(sampled.log_prob.item()),
            logp_action=float(self.model.action_log_prob(input_tensor,sampled.raw_action).item()),
            value=float(sampled.value.item()))

    def append(self, row):
        if row['version'] != self.version or row['sample_serial'] != self.version*STEPS+self.position:
            raise ValueError('A distinct timing transition cannot enter a different policy version.')
        row['collector_position'] = self.position
        row['collector_cut'] = self.position+1 == STEPS
        jsonline(batch_path(self.run,self.version,self.case), row)
        self.position += 1
        if self.position == STEPS:
            atomic_json(self.run/f'ready-{self.version:03d}-{self.case}.json',dict(
                        version=self.version,count=self.position,case=self.case,
                        last_decision=row['decision_id'],last_physical_s=row['end_s']))


def close_interrupted_sequence(run, case, attempt, reason):
    """Terminate GAE at the last actual saved successor, without creating another transition."""
    version = json.loads((run/'published.json').read_text())['version']
    rows = read_rows(batch_path(run,version,case))
    if not rows:
        return
    last = rows[-1]
    if (last['attempt'] != attempt or last['terminated'] or last['truncated']
            or last['collector_cut']):
        return
    target = run/f'interrupted-{version:03d}-{case}.jsonl'
    if any(event['decision_id']==last['decision_id'] for event in read_rows(target)):
        return
    jsonline(target,dict(attempt=attempt,decision_id=last['decision_id'],version=version,case=case,
        physical_s=last['end_s'],next_value=last['next_value'],reason=reason,
        interpretation='Unfinished later request excluded. Last already-saved successor bootstraps; GAE stops.'))


def duration_gae(reward, value, next_value, terminated, truncated, cut, dt):
    gamma = torch.pow(.99, dt/.5)
    lam = torch.pow(.95, dt/.5)
    delta = reward + gamma * (~terminated) * next_value - value
    continuation = ~(terminated | truncated | cut)
    advantage = torch.empty_like(reward)
    carry = torch.zeros_like(reward[-1])
    for i in reversed(range(reward.shape[0])):
        carry = delta[i] + gamma[i]*lam[i]*continuation[i]*carry
        advantage[i] = carry
    return advantage, advantage+value


def train_batch(run, version, head, academic_identity, device='cuda'):
    model, _ = restore(checkpoint_path(run,version),device,optimizer=True)
    streams = [read_rows(batch_path(run,version,case)) for case in SPEC['cases']]
    for case,stream in zip(SPEC['cases'],streams):
        actual_cuts = {r['decision_id'] for r in read_rows(run/f'interrupted-{version:03d}-{case}.jsonl')}
        for row in stream:
            row['interrupted_cut'] = row['decision_id'] in actual_cuts
    rows = [row for stream in streams for row in stream]
    if [len(stream) for stream in streams] != [STEPS,STEPS] or len({row['decision_id'] for row in rows}) != 2*STEPS:
        raise ValueError('A PPO update requires two disjoint 128-transition collector streams.')
    for case, stream in zip(SPEC['cases'],streams):
        for i,row in enumerate(stream):
            if (row['case'],row['head'],row['version'],row['collector_position'],row['sample_serial']) != (
                    case,head,version,i,version*STEPS+i) or not row['dt_s']>0:
                raise ValueError('Invalid collector identity, serial order, or elapsed physical time.')
            if i and stream[i-1]['attempt'] != row['attempt'] and not (
                    stream[i-1]['terminated'] or stream[i-1]['truncated'] or stream[i-1]['interrupted_cut']):
                raise ValueError('An episode boundary must close the preceding GAE chain.')
    def field(name, dtype=torch.float32):
        return torch.as_tensor(np.asarray([[s[i][name] for s in streams] for i in range(STEPS)]),
                               dtype=dtype,device=device)
    obs = field('obs')
    raw = field('raw_action')
    value = field('value')
    with torch.no_grad():
        advantages, returns = duration_gae(field('reward'),value,field('next_value'),
            field('terminated',torch.bool),field('truncated',torch.bool) | field('interrupted_cut',torch.bool),field('collector_cut',torch.bool),
            field('dt_s'))
    batch = Batch(obs=obs.flatten(0,1), act=raw.reshape(-1,1), logp_old=field('logp_raw').reshape(-1),
        v_s=value.reshape(-1), adv=advantages.reshape(-1), returns=returns.reshape(-1), info=Batch())
    with torch.no_grad():
        logp_error = (model.policy(batch).dist.log_prob(batch.act)-batch.logp_old).abs().max().item()
        physical_logp = model.action_log_prob(batch.obs,batch.act)
        physical_error = (physical_logp-field('logp_action').reshape(-1)).abs().max().item()
        action_error = (raw.tanh()-field('action')).abs().max().item()
        if logp_error > 5.e-5 or physical_error > 5.e-5 or action_error > 1.e-6:
            raise ValueError('Behavior request action or log-density differs from this on-policy model.')
    wall = time.perf_counter()
    cpu = time.process_time()
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    model.policy.updating = True
    try:
        losses = model.policy.learn(batch,batch_size=SPEC['ppo']['minibatch'],repeat=SPEC['ppo']['epochs'])
    finally:
        model.policy.updating = False
    with torch.no_grad():
        log_ratio = model.policy(batch).dist.log_prob(batch.act) - batch.logp_old
        ratio = log_ratio.exp()
        kl = (ratio-1-log_ratio).mean().item()
        clip_fraction = ((ratio-1).abs()>SPEC['ppo']['clip']).float().mean().item()
    metrics = dict(version_input=version,version_output=version+1,head=head,
        cumulative_distinct=(version+1)*STEPS*2,distinct=2*STEPS,gae_shape=list(advantages.shape),
        end_rewards_finite=bool(torch.isfinite(returns).all().item()),
        min_dt_s=min(row['dt_s'] for row in rows),max_dt_s=max(row['dt_s'] for row in rows),
        simulated_event_seconds=sum(row['dt_s'] for row in rows),old_actor_updates=0,
        requested_gradient_rows=2*STEPS,effective_liftoff=sum(bool(r['execution'] and r['execution']['effective']) for r in rows),
        only_ramp=sum(bool(r['execution'] and r['execution']['transfer_fraction_changed'] and not r['execution']['effective']) for r in rows),
        unexecuted=sum(r['execution'] is None for r in rows),episode_terminals=sum(r['terminated'] for r in rows),
        time_limit_truncations=sum(r['truncated'] for r in rows),technical_sequence_cuts=sum(r['interrupted_cut'] for r in rows),collector_cuts=2,
        kl=kl,clip_fraction=clip_fraction,behavior_max_logp_error=logp_error,physical_max_logp_error=physical_error,action_max_error=action_error,
        losses={k:float(np.mean(v)) for k,v in losses.items()},optimizer_steps=len(losses['loss']),
        advantage_mean=float(advantages.mean().item()),return_mean=float(returns.mean().item()),
        optimization_wall_s=time.perf_counter()-wall,optimization_cpu_s=time.process_time()-cpu,
        device=device,peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None)
    if not metrics['end_rewards_finite']:
        raise ValueError('Non-finite event return.')
    # Exact pre-flatten advantages and masks; the ledger independently stores physical successors.
    np.savez_compressed(run/f'credit-{version:03d}.npz',advantage=advantages.cpu().numpy(),
        returns=returns.cpu().numpy(),dt_s=field('dt_s').cpu().numpy(),terminated=field('terminated',torch.bool).cpu().numpy(),
        truncated=field('truncated',torch.bool).cpu().numpy(),interrupted=field('interrupted_cut',torch.bool).cpu().numpy(),cut=field('collector_cut',torch.bool).cpu().numpy())
    save_policy(checkpoint_path(run,version+1),model,version+1,head,academic_identity,metrics)
    atomic_json(run/f'update-{version+1:03d}.json',metrics)
    return metrics

