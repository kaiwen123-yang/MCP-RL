# W38 / S5-r27 controller audit — 2026-11-05

## Scope and conclusion

- Reviewed repository head: `5e00beca0896bb5517bab8c1dcc441303e5c2098`.
- Implementation comparison: `fc322962` to the reviewed head.
- Source snapshot: [MCP-RL at the reviewed head](https://github.com/kaiwen123-yang/MCP-RL/tree/5e00beca0896bb5517bab8c1dcc441303e5c2098). Paths below are relative to the repository root; line references describe that snapshot.
- Read the actual r27 patch, parent r26 patch, controller/model/runtime integration, PPO terminal settlement, build identities, and both new E30 compact results/resolved configurations. This is not a claim to have reread every historical file.
- No runtime/controller source changes, SSH, build, physics, controller replay, or new experiment were performed for this review. The audit and execution-status metadata are documentary updates.
- New r27 raw NPZ, `planner-records.jsonl`, and `feedback-actions.jsonl` remain on the server. Their contents were **not available to this review**. The old r26 raw data are not evidence of the new r27 event order.

**Conclusion:** r27 implemented the intended C2 approach activation and physical-terminal reward settlement. No demonstrated wiring error in these changes explains the two failures. Plant and MPC already share PD/actuator saturation; it would be false to prescribe a missing torque limiter. Candidate task feasibility and the validity of the local feedback/contact approximation remain distinct, plausible failure mechanisms. Choose the next single revision from a short extraction of the existing two E30 records; do not assume that a large new K term or an unsafe accepted rollout has already been observed.

## 1. Verified implementation and actual configuration

### 1.1 C2 cost gate is present in both intended residuals

`experiments/terrain_memory/s1-go2-r11-S5-r27.patch:7–32` implements the clipped quintic smoothstep and

\[
g(x,y)=S((x-x_{min}+b)/b)S((x_{max}+b-x)/b)S((y_{max}+b-|y|)/b).
\]

`r27.patch:38–60` removes the old spatial Boolean from LandingVelocity and multiplies its existing residual by the same helper. ImpactMargin uses the complete `g*r` in `R26ImpactValue`. The parent `r26.patch:62–102,267–339` calls that function for active-substep selection and geometry FD, then composes with the original P/Q prefixes. Terminal risk uses the same function. The original ten A/B transitions and node-start full C/D remain.

There is no separate, still-hard candidate cost using the old approach Boolean. LandingIndentation retains its original true-material Boolean. True contact/filter boundaries are unchanged. The C2 property applies to the artificial spatial gate, not to all contacts, positive-part functions, or substep max switches.

Actual numeric values from both E30 `*-resolved.json`, rather than old source comments:

| Quantity | Actual value |
|---|---:|
| Approach band | 0.07525 m |
| Instantaneous MPC local contact stiffness | 1,227,272.7272727273 N/m |
| Impact onset / scale | 0.0008 / 0.0002 m |
| TrunkClearance scale | 0.001 m |
| Prediction integration / input hold | 0.0002 s × 10 |
| Reference horizon / lead bound / speed bound / acceleration | 0.35 s / 0.15 m / 0.25 m/s / 0.5 m/s² |
| Moving gait | Trot, 2 Hz, duty 0.45, amplitude 0.005 m |
| MakeDifferentiable | enabled, inherited |
| r25 MovingSupport configuration | `r12_stable_support` absent |

The `375000` comment beside the fixed LandingIndentation onset is not the instantaneous MPC stiffness. It corresponds to the equilibrium TM02 stiffness; the distinction is intentional in the inherited contract, as detailed below.

### 1.1a Do not mistake the K0/K-infinity distinction for a new r27 bug

TM02 parameters are `K_eq=500`, `G_eq=300`, `K_ov=1000`, `G_ov=700` MPa. The original material setup uses gamma=0.025, full foot area=0.001 m² and thickness=0.05 m (`experiments/tm02/r2-multihistory.json`; `src/tm02/s1_contact.py:18–23`). From the unconstrained-lateral uniaxial relation

\[
E=9KG/(3K+G),\qquad k=\gamma(A/H)E\,10^6,
\]

the frozen-internal-state instantaneous values are E0=2454.5454545 MPa and k0=1,227,272.7273 N/m; the relaxed values are E-infinity=750 MPa and k-infinity=375,000 N/m. The corresponding two-foot equilibrium indentation scales are 60.77494397 µm and 198.89981664 µm.

- **LandingIndentation is active**, not dead code: weight 0.1, 1 mm scale, and `positive_part(-z - 0.00019889981664)`. It is a one-sided penalty past that indentation, not an equality commanding the foot to reach 198.9 µm.
- **The loaded velocity reference already uses k0 correctly**: `loaded_target=rest-mg/(2*normal_stiffness[0])`. The actual insertion is in `scripts/prepare_s1_go2_r11.py:128–148` / the executed r21 patch and is inherited through the r24/r26 base. It does not use the fixed 198.9 µm constant.
- `experiments/terrain_memory/s1-go2-r11-S5-r21.json:r11_contact_completion.scope`, also copied into the current E30 resolved configuration, explicitly preserves the K-infinity-based indentation penalty while using K0 in the observed-rest loaded reference.
- **MaterialSupportHeight is active**, weight 1, scale 0.04 m, fixed absolute base target 0.29020336039643685 m. It derives from the retained physical stance, not from either indentation constant. It remains active throughout moving gait; Stand uses the existing separate stance reference.
- The historical S5-r1 preparation already changed both indentation threshold and normalization to K0 (`scripts/prepare_s1_go2_r11.py:45–53,225–229`); the subsequent S5-r3 branch explicitly did not stack that r1/r2 formulation (`:231–237`). Do not present reapplying it as a newly discovered repair.

K0 is a declared instantaneous linear proxy, not the exact algorithmic tangent of every real 0.2 ms material step. `ReferenceColumns.step` solves history-dependent internal strain and lateral stress together (`src/tm02/s1_material.py:16–64`); the tangent after that implicit solve can depend on dt, state and stress-dependent viscosity. This is an existing model approximation. No material solver or physical step was run to derive the algebraic constants above.

### 1.2 Terminal settlement occurs before the final transition

`scripts/s1_go2_r11_policy.py:87–94` adds `settle_physical_terminal`. `:475–489` identifies the real physical failure, performs the final route update, settles incomplete passage credit, then forms the final observation/transition. External collection truncations do not take this path.

Both E30 records contain a settlement to legal progress zero. Independent full-result accounting by the experiment-history audit found all 68 naturally failed new training episodes settled, while the two target truncations preserve their separate semantics. The low-frequency transition/rollout raw files are not in Git, so this controller audit does not claim an independent raw-transition recomputation.

### 1.3 Actual outcome, not startup metadata

At the reviewed head, the r27 spec/AGENTS still contained `joint_sampling_running` startup wording. The later actual job records show completed collection/evaluation; the r27 execution-status metadata is updated alongside this published audit.

| E30 | End time | End reason | Maximum x | Legal outbound/return |
|---|---:|---|---:|---|
| w38-a, seed-directory 1 | 13.5634 s | trunk geom 13 contact | 0.6067477656 m | 0 / 0 |
| w38-b, seed-directory 2 | 13.5208 s | trunk geom 13 contact | 0.7927774383 m | 0 / 0 |

Both loaded material with a foot, but neither body entered x=0.8 in E30. All 12 joints reached their configured force limit at least once. This is a maximum-over-time observation, not a saturation duration, frequency, onset time, or proof that feedback caused saturation.

Each branch actually collected +1024 transitions and +4 updates from the same ep216 parent, ending independently at 7569 / 29, tail 0, excluded history 145. They are not one policy with eight new updates.

## 2. PD, control bounds, and derivatives: what is already consistent

`scripts/s1_go2_r11_model.py:91–105,138–142` builds the same unit-gear position-command actuator model for plant and Agent:

\[
\tau_{req}=60(u-q)-5\dot q,\qquad
\tau=\operatorname{clip}_{[\tau_{min},\tau_{max}]}(\tau_{req}).
\]

The actuator position-command range is the original joint-angle range. Force limits are ±23.7 Nm for hip/thigh and ±45.43 Nm for calf.

- Candidate `ActionRollouts` clips the position target before calling the model rollout: retained `mjpc/planners/ilqg/planner.cc:637–694`.
- Execution policy computes the same affine feedback then clips to the same control range: retained `policy.cc:151–160`, with the r27 recording-only wrapper.
- The rollout model and all ten FD transitions contain the same PD and MuJoCo actuator force saturation. `MakeDifferentiable` changes contact/joint solimp d0, not actuator force limits: retained `mjpc/utilities.cc:59–75` and `agent.cc:299–318`.
- `scripts/s1_go2_r11_runtime.py:539–557,571–616` applies and holds the target for ten physical substeps, records actual actuator force, and records the raw PD request.
- Backward-pass BoxQP explicitly constrains position commands: retained `backward_pass.cc:158–193`; `planner.cc:453–483` passes `model->actuator_ctrlrange`.

**Therefore there is no demonstrated missing-forcerange or extra-execution-only clipping bug.** There is a different structural limitation: the local QP's explicit box is a fixed angle-target box. State-dependent torque saturation and contact active sets enter implicitly through the nonlinear dynamics/FD; the deployed affine K does not have an explicit contact-validity domain.

### Why instantaneous target projection is not a demonstrated repair

At a measured state, the nonsaturating PD target interval is

\[
L_i(x)=\max(u_{min,i},q_i+(5\dot q_i+\tau_{min,i})/60),\quad
U_i(x)=\min(u_{max,i},q_i+(5\dot q_i+\tau_{max,i})/60).
\]

When this intersection is nonempty, projecting a target to this interval produces the **same instantaneous torque** as the already-present force limiter. It can change later torque over the held 2 ms interval; that cannot be inferred from a one-state algebraic counterfactual. At high velocity near a joint limit the intersection may even be empty.

Thus a new final-output clip, arbitrary K scaling, or low-pass filter is not justified by the compact maxima. If explicit state-dependent actuator constraints are selected after inspecting the event, they must be part of local optimization, candidate-feedback rollouts, and execution. Active torque-bound rows have nonzero state derivatives (position plus kd/kp times velocity); pretending they are fixed control-box rows with zero active feedback is incorrect. Whole-hold torque feasibility additionally depends on all ten state transitions, not just the first state.

## 3. Nonfoot support is an allowed route through the current objective

This is a source fact, not proof of what the new accepted trajectories selected:

- Runtime uses the original allowed leg-body list; trunk/world contacts terminate: `scripts/s1_go2_r11_runtime.py:110–134,236–241,617–619`; model construction `scripts/s1_go2_r11_model.py:73–75`.
- TrunkClearance enumerates collision geoms belonging to the trunk body. The domain, landing, and impact rows cover the four feet: actual `r26.patch:12–60`.
- No direct knee/shin/thigh contact-force or nonfoot-clearance residual/constraint was added by r27.
- Height/Upright/Balance/Posture are soft objectives. Balance uses average foot position, not the actual supporting contact set. Effort penalizes actuator force, so support supplied by passive leg-body contact is not directly charged as a forbidden support mode: retained `quadruped.cc:105–130,197–214`.

It is consequently possible in the optimizer model to shift support toward allowed leg geometry, reduce foot-only risk/effort, and still pay finite pose/height costs. Old r26 raw data demonstrated actual leg support, but **that old event order cannot be asserted for r27**. New compact records do not show whether nonfoot contact preceded large feedback, followed it, or was already present in the accepted prediction.

Changing the real task to terminate on any leg contact would change the authorized evaluation and is not the remedy. A future foot-only planned-contact restriction would be a declared controller constraint, not an unchanged interpretation of the original task. Reinstalling r25 force/load allocation is not supported.

## 4. New diagnostics are implemented, but their coverage has limits

### What the actual C++ writes

`r27.patch:89–145` writes RPC feedback samples with `time`, `plan_id`, `feedback_scaling`, `nominal_unclipped`, tangent-space `state_delta`, `raw_K_delta_x`, `action_before_clip`, and `action_after_clip`.

`r27.patch:170–244` writes successful plan updates with winner, regularization before/used/after, line-search step, accepted cost, feedback-reference cost, and role identities. When current base x is in [0.4,1.2] or [1.8,2.6], it includes node 0 and node 10 (+20 ms): linearization state/residual, accepted state/action, feedback-reference state/nominal, K, and LandingVelocity/ImpactMargin C/D.

The environment variables are passed to the actual training Agent (`scripts/train_s1_go2_r12.py:124–130`) and evaluation Agent (`scripts/s1_go2_r11_runtime.py:348–360`). This is more than a planned, unused `.inc` file.

### What is not established by the available Git evidence

- Git contains only diagnostic path/semantics declarations in resolved JSON, not those two JSONL contents or a derived numerical summary. The finisher intentionally retains raw JSONL; see `scripts/finish_s1_go2_r12_job.py:161–214`.
- No r27 raw K-delta maximum, first clipping event, selected C/D spike, or accepted collision can be claimed from the current compact package.
- A saved accepted state violating a boundary at node 0 or 10 proves a violation at that node. Two safe endpoints do **not** establish feasibility of intervening 0.2 ms substeps or the remaining 0.35 s horizon.
- Full prediction NPZ still comes from `BestTrajectory() == policy.trajectory`; it is the feedback reference, not a separately recorded full accepted rollout. Do not fill missing accepted nodes by interpolation and call them measured predictions.
- Plan records are written after successful policy updates. A backward-pass failure/early return has no equivalent successful plan row; an unchanged `plan_id` can represent execution of an older policy.
- Action records are RPC samples. Startup requests at time zero include intermediate diagnostic policies not integrated into the plant; join with the actual trajectory before calling them executed actions.
- Risk sensor row order is FL, RL, FR, RR; physical runtime foot arrays are FL, FR, RL, RR. Respect these mappings.
- The appended final trajectory row can have a newer qpos/qvel time than its retained action/diagnostic timestamp. Exclude mismatched timestamps from feedback comparisons.

## 5. Minimum existing-data extraction for the next session

Do this before choosing the next controller mechanism, within the same interactive budget. It is read-only analysis of already executed E30s, not a new physical qualification, benchmark, replay, or isolated module experiment.

Read each of:

`outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261104T104730Z-w38-a-eval-PANN_PPO_S5-s1-E30/`

`outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261104T104730Z-w38-b-eval-PANN_PPO_S5-s2-E30/`

### Locate event times by scanning existing records

After the actual Stand→outbound transition, identify:

1. First actuator-target clipping (`action_before_clip` versus `action_after_clip`), and first raw-PD saturation at the configured force limit. Keep these separate.
2. First |raw K-delta| > 0.5 rad as a descriptive comparison with W37; this is not a new physical stop criterion or tuning threshold.
3. First nonfoot contact count and its actual nonfoot Fz; also report its support fraction over the retained local window, rather than treating a vanishing contact as a large support transfer.
4. First real material `load` event from `material-events.jsonl`.
5. The already-recorded physical terminal time.

Retain and merge windows [event−0.10 s, event+0.20 s], clipped to the recorded episode, plus the final 0.10 s. This selects roughly one to two seconds per episode, not whole bulk archives. Scan the old files to find times; do not sample new trajectories. If a required source file is absent, record that limitation rather than manufacturing a value.

### Minimum retained fields

- Original clocks: physical force-evaluation and post-integration time, control time, diagnostic time, plan origin/id, and actual phase.
- Measured qpos/qvel; executed target; original ctrlrange, forcerange, kp/kd; actual torque and raw PD request; foot and nonfoot forces/contact counts; full foot xyz/velocity; current causal anchor userdata and reference packet.
- Complete action-record fields above. Join actual action_after_clip to the trajectory target, allowing only storage precision differences.
- Plan records for the overlapping plan ids, retaining only existing slices. Keep linearization/accepted/feedback-reference roles separate, K/C/D as stored, winner, action step, regularization and costs.
- Exact material load event and original terminal result; no new contact criterion.
- Use the current model geometry for purely kinematic foot/leg/trunk gap calculations. No integration or material recomputation is required.

At each retained measured state calculate both

\[
\tau_{nom}=\operatorname{clip}_\tau(60(\operatorname{clip}_u(u_{nom})-q)-5\dot q),\qquad
\tau_{exec}=\operatorname{clip}_\tau(60(u_{exec}-q)-5\dot q).
\]

This distinguishes a nominal command already at saturation from feedback moving it into a different saturation set. It does not establish a different counterfactual next state. Also report full tangent q/v errors; a small qpos error alone does not prove an abnormal gain.

## 6. Conditional priority for one subsequent complete-joint revision

### First distinguish two mechanisms

**A. The accepted candidate already enters task-terminal geometry.** A violation in the recorded accepted slices is sufficient evidence for that point. This supports a candidate-feasibility/acceptance repair. It does not support arbitrary damping of K.

**B. Available accepted slices are geometrically admissible, but the measured feedback changes actuator/contact active sets before the observed failure.** This supports examining the domain of the local feedback model. Absence of an accepted violation in two slices is not proof that the full accepted horizon is safe. Use available time/identity alignment honestly; do not elevate the missing horizon to evidence.

If neither can be supported, report insufficient causal evidence and the exact missing record. Do not launch r27 with a new name or choose K clipping/force cost just to fill the schedule.

### Priority candidate if A is established: prevent cost from purchasing a terminal trunk contact

Using the same original trunk geoms/margins already used by TrunkClearance, define for each actual nonlinear candidate rollout

\[
v_c=\max_{t,s,j}\,[m_j-d_j(x_{t,s})]_+,
\]

including all ten pre-integration substeps and the terminal state. Among geometrically admissible candidates, select the minimum original task cost J. Do not replace the existing cost with a larger arbitrary trunk weight and call that a hard constraint.

The parent already caches componentwise substep peak TrunkClearance. Its monotone positive-part expression can expose the underlying original gap, or a small explicit maximum-gap field can be accumulated at the same place. `BestRollout`, both nominal and action rollout selection, and accepted-plan summaries must share one feasibility definition. Do not test a new reduced proxy only after execution.

Necessary limitations/implementation work:

- Existing soft-trunk derivatives can guide candidate generation, but selecting feasible members of a finite line search is **not** a complete constrained DDP solver or a true-plant safety proof.
- If no candidate is feasible, report that state and an explicit restoration policy; do not freeze an old action and label it safe. Feasibility restoration/acceptance must be consistent with regularization/improvement accounting. Simply swapping the winner while continuing to interpret cost-only surprise as feasible progress is incomplete.
- The nonlinear model still differs from the plant. This experiment addresses accepted-model task semantics, not all physical prediction error.
- Allowed leg-body contacts remain an independent objective freedom. Do not silently expand the real failure set. If the evidence instead points specifically to planned leg support, its remedy needs an explicit controller contact-mode constraint and its derivatives, not reinstatement of r25 load-sharing residuals.

### If B is established instead

The appropriate target is a local state-dependent actuation/contact feasible domain. At a minimum the same PD feasible inequalities must be used in optimization, candidate feedback, and execution; active-bound derivatives and all-hold dynamics must be handled. This is a different revision from candidate task-feasibility selection, and should not be stacked into the same uninformative trial. The current audit does not prescribe arbitrary gain caps, new filters, or a paper's steady-state torque formula as a replacement for this Go2's transient dynamics.

After selecting one evidence-backed revision, use the authorized normal build and complete joint streams directly. Keep the C2 repair, terminal settlement, original physical/evaluation definitions, and existing compact delivery. No additional pure-MPC entry test is proposed.

## 7. Research connection and scope

[Contact Trust Region, arXiv:2505.02291v4](https://arxiv.org/html/2505.02291v4), particularly equations 24/26 and section 3.6.2, motivates constraining a local approximation by contact/actuator feasibility instead of assuming a symmetric perturbation region is physically valid. Its local subproblem is an SOCP built on a quasidynamic contact model; the authors explicitly limit the torque constraint to steady state. It does not demonstrate dynamic Go2 stability, justify arbitrary K clipping, or supply the missing force derivatives for the present MuJoCo/TM02 model. The proposed candidate-acceptance investigation is a project-specific application of the feasibility principle, not a reproduction of that algorithm.

## Main fixed-head evidence paths

- `experiments/terrain_memory/s1-go2-r11-S5-r27.patch`
- `experiments/terrain_memory/s1-go2-r11-S5-r26.patch`
- `experiments/terrain_memory/s1-go2-r11-S5-r27.json`
- `scripts/s1_go2_r11_model.py`
- `scripts/s1_go2_r11_runtime.py`
- `scripts/s1_go2_r11_policy.py`
- `scripts/train_s1_go2_r12.py`
- `experiments/terrain_memory/s1-go2-r25-source/mjpc/planners/ilqg/{planner,policy,backward_pass}.cc` (shared inherited sections; r27 recording additions are in its actual patch)
- `experiments/terrain_memory/s1-go2-r25-source/mjpc/tasks/quadruped/quadruped.cc` (shared inherited objective sections; do not treat the removed r25 MovingSupport branch as current r27 code)
- `results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261104T104730Z-w38-a-eval-PANN_PPO_S5-s1-E30{,-resolved}.json`
- `results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261104T104730Z-w38-b-eval-PANN_PPO_S5-s2-E30{,-resolved}.json`
- Both `results/.../jobs/job-20261104T104730Z-w38-{a,b}/training/{collection,learning,stream-identity}.json` and retained controller identities.
