# Legacy Code to Remove

Files no longer used by `train_distributed.py` or `training_dashboard.py`.

## Summary

| Category | Files to Remove |
|----------|----------------|
| `mario_rl/training/` | 13 files |
| `mario_rl/buffers/` | 4 files (entire dir) |
| `mario_rl/core/` | 8 files |
| `mario_rl/ui/` | 3 files (entire dir) |
| `mario_rl/utils/` | 2 files (entire dir) |
| `mario_rl/environment/` | 2 files |
| `mario_rl/distributed/` | 2 files |
| `mario_rl/agent/` | 3 files |
| `scripts/` | 12 files |

**Total: ~49 files to remove**

---

## Detailed List

### 1. `mario_rl/training/` (legacy workers)

| File | Reason |
|------|--------|
| `appo_learner.py` | Legacy APPO algorithm |
| `appo_worker.py` | Legacy APPO algorithm |
| `ddqn_learner.py` | Replaced by `learners/ddqn.py` |
| `ddqn_status.py` | Used only by legacy `ddqn_worker.py` |
| `ddqn_worker.py` | Replaced by `distributed/training_worker.py` |
| `learner.py` | Legacy learner |
| `ppo_learner.py` | Legacy PPO algorithm |
| `ppo_worker.py` | Legacy PPO algorithm |
| `rollout_buffer.py` | Used only by PPO |
| `shared_buffer.py` | Legacy shared buffer |
| `shared_gradients.py` | Replaced by `shared_gradient_tensor.py` |
| `worker.py` | Legacy worker |
| `world_model_learner.py` | Legacy world model |

**Keep:**
- `shared_gradient_tensor.py` (used by distributed)
- `training_ui.py` (used by distributed)
- `snapshot.py` (used by snapshot_handler)
- `snapshot_handler.py` (used by snapshot_wrapper)
- `snapshot_state_machine.py` (used by snapshot_handler)

### 2. `mario_rl/buffers/` (entire directory)

| File | Reason |
|------|--------|
| `__init__.py` | Part of legacy buffers |
| `sum_tree.py` | Used only by legacy `prioritized.py` |
| `prioritized.py` | Replaced by `core/replay_buffer.py` |
| `nstep.py` | Used only by legacy workers |

### 3. `mario_rl/core/` (legacy DDQNWorker helpers)

| File | Reason |
|------|--------|
| `episode.py` | Used only by legacy DDQNWorker |
| `exploration.py` | Used only by legacy DDQNWorker |
| `metrics.py` | Used only by legacy DDQNWorker |
| `timing.py` | Used only by legacy DDQNWorker |
| `ui_reporter.py` | Used only by legacy DDQNWorker |
| `weight_sync.py` | Used only by legacy DDQNWorker |
| `elite_buffer.py` | Used only by legacy DDQNWorker |
| `reward_normalizer.py` | Used only by legacy DDQNWorker |

**Keep:**
- `config.py` (used by snapshot_wrapper)
- `device.py` (used by train_distributed)
- `env_runner.py` (used by training_worker)
- `replay_buffer.py` (used by training_worker)
- `types.py` (used everywhere)

### 4. `mario_rl/ui/` (entire directory)

| File | Reason |
|------|--------|
| `__init__.py` | Legacy |
| `metrics.py` | Legacy |
| `training_ui.py` | Duplicate, actual one is in `training/` |

### 5. `mario_rl/utils/` (entire directory)

| File | Reason |
|------|--------|
| `__init__.py` | Not used |
| `metrics.py` | Not used |

### 6. `mario_rl/environment/` (partial)

| File | Reason |
|------|--------|
| `env_factory.py` | Duplicate of `factory.py` |
| `frame_stack.py` | Gymnasium's `FrameStackObservation` is used instead |

**Keep:**
- `factory.py` (used by snapshot_wrapper)
- `mariogym.py` (used by factory)
- `snapshot_wrapper.py` (used by train_distributed)
- `wrappers.py` (used by factory)

### 7. `mario_rl/distributed/` (legacy)

| File | Reason |
|------|--------|
| `coordinator.py` | Replaced by `training_coordinator.py` |
| `worker.py` | Replaced by `training_worker.py` |

**Keep:**
- `training_coordinator.py`
- `training_worker.py`
- `events.py`
- `shm_gradient_pool.py`
- `shm_heartbeat.py`

### 8. `mario_rl/agent/` (partial)

| File | Reason |
|------|--------|
| `neural.py` | Legacy `DuelingDDQNNet`, not used by visualization |
| `ppo_net.py` | Legacy PPO network |
| `replay.py` | Legacy replay buffer |

**Keep (for visualization scripts):**
- `ddqn_net.py` (used by visualization scripts)
- `world_model.py` (used by visualization scripts)

### 9. `scripts/` (legacy training scripts)

| File | Reason |
|------|--------|
| `train_ddqn_distributed.py` | Replaced by `train_distributed.py` |
| `train_ddqn_simple.py` | Legacy single-process training |
| `train_ppo_simple.py` | Legacy PPO |
| `train_ppo_custom.py` | Legacy PPO |
| `train_ppo.py` | Legacy PPO |
| `train_appo.py` | Legacy APPO |
| `train_modular.py` | Legacy |
| `train.py` | Legacy |
| `train.sh` | Legacy shell script |
| `watch.py` | Uses legacy agent files |
| `benchmark_shared_gradients.py` | Uses legacy shared_gradients |
| `benchmark_worker_ops.py` | Uses legacy worker |

**Keep:**
- `train_distributed.py` (main training script)
- `training_dashboard.py` (dashboard)
- `plot_training.py` (plotting utility)
- `train_runpod.sh` (deployment shell wrapper)
- `visualize_world_model_dynamics.py` (diagnosis)
- `visualize_world_model_reconstruction.py` (diagnosis)
- `diagnose_decoder_mode_collapse.py` (diagnosis)

---

## Command to Remove All

```bash
# From project root
rm -rf mario_rl/buffers/
rm -rf mario_rl/ui/
rm -rf mario_rl/utils/

rm mario_rl/training/appo_learner.py
rm mario_rl/training/appo_worker.py
rm mario_rl/training/ddqn_learner.py
rm mario_rl/training/ddqn_status.py
rm mario_rl/training/ddqn_worker.py
rm mario_rl/training/learner.py
rm mario_rl/training/ppo_learner.py
rm mario_rl/training/ppo_worker.py
rm mario_rl/training/rollout_buffer.py
rm mario_rl/training/shared_buffer.py
rm mario_rl/training/shared_gradients.py
rm mario_rl/training/worker.py
rm mario_rl/training/world_model_learner.py

rm mario_rl/core/episode.py
rm mario_rl/core/exploration.py
rm mario_rl/core/metrics.py
rm mario_rl/core/timing.py
rm mario_rl/core/ui_reporter.py
rm mario_rl/core/weight_sync.py
rm mario_rl/core/elite_buffer.py
rm mario_rl/core/reward_normalizer.py

rm mario_rl/environment/env_factory.py
rm mario_rl/environment/frame_stack.py

rm mario_rl/distributed/coordinator.py
rm mario_rl/distributed/worker.py

rm mario_rl/agent/neural.py
rm mario_rl/agent/ppo_net.py
rm mario_rl/agent/replay.py

rm scripts/train_ddqn_distributed.py
rm scripts/train_ddqn_simple.py
rm scripts/train_ppo_simple.py
rm scripts/train_ppo_custom.py
rm scripts/train_ppo.py
rm scripts/train_appo.py
rm scripts/train_modular.py
rm scripts/train.py
rm scripts/train.sh
rm scripts/watch.py
rm scripts/benchmark_shared_gradients.py
rm scripts/benchmark_worker_ops.py
```
