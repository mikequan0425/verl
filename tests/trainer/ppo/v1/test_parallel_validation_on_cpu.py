# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only ordering tests for parallel validation in V1 async trainers."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode, PPOTrainerSeparateAsync


class _FinishedValidationFuture:
    def __init__(self, events):
        self.events = events

    def result(self):
        self.events.append("validation_drain")
        return {"val/test_score": 1.0}


def _pending_validation_state(trainer, events):
    trainer._parallel_validation_executor = None
    trainer._parallel_validation_future = _FinishedValidationFuture(events)
    trainer._parallel_validation_step = 7
    trainer._parallel_validation_start_time = 0.0
    trainer._replay_buffer_lock = threading.Lock()
    trainer._parallel_validation_started_event = threading.Event()
    trainer._parallel_validation_started_event.set()
    trainer.logger = SimpleNamespace(log=lambda data, step: events.append(("validation_log", step)))


def test_wait_logs_metrics_at_the_captured_validation_step():
    events: list = []
    trainer = PPOTrainerColocateAsync.__new__(PPOTrainerColocateAsync)
    _pending_validation_state(trainer, events)

    metrics = trainer._wait_parallel_validation()

    assert metrics["val/test_score"] == 1.0
    assert "parallel_validation/elapsed_time" in metrics
    assert events == ["validation_drain", ("validation_log", 7)]
    assert trainer._parallel_validation_future is None


def test_colocate_async_drains_validation_before_reclaiming_rollout():
    events: list = []
    trainer = PPOTrainerColocateAsync.__new__(PPOTrainerColocateAsync)
    _pending_validation_state(trainer, events)
    trainer.curr_step_profile = False
    trainer.checkpoint_manager = MagicMock()
    trainer.checkpoint_manager.abort_replicas.side_effect = lambda: events.append("abort")
    trainer.checkpoint_manager.sleep_replicas.side_effect = lambda: events.append("sleep")

    trainer.on_sample_end()

    assert events == ["validation_drain", ("validation_log", 7), "abort", "sleep"]
    trainer.checkpoint_manager.abort_replicas.assert_called_once()
    trainer.checkpoint_manager.sleep_replicas.assert_called_once()


def test_separate_async_keeps_hybrid_in_rollout_until_validation_drains():
    events: list = []
    trainer = PPOTrainerSeparateAsync.__new__(PPOTrainerSeparateAsync)
    _pending_validation_state(trainer, events)
    trainer.current_mode = HybridEngineMode.ROLLOUT
    trainer.timing_raw = {}
    trainer.hybrid_rollout_config = SimpleNamespace(enable_switch=False)
    trainer.replay_buffer = MagicMock()
    trainer.replay_buffer.wait_for_sampleable.side_effect = lambda *args, **kwargs: (
        events.append("rollout_wait")
        or (
            set(),
            {},
        )
    )
    trainer._step_threshold = 1

    def switch_to_trainer():
        events.append("switch_to_trainer")
        trainer.current_mode = HybridEngineMode.TRAINER

    trainer.switch_to_trainer = switch_to_trainer

    trainer.on_step_begin()
    assert trainer.current_mode == HybridEngineMode.ROLLOUT
    trainer.replay_buffer.get_sampleable_count.assert_not_called()

    trainer._wait_for_sampleable_and_switch()

    assert events == ["validation_drain", ("validation_log", 7), "rollout_wait", "switch_to_trainer"]
    assert trainer.current_mode == HybridEngineMode.TRAINER


def test_separate_async_preserves_inventory_gate_while_validation_is_pending():
    events: list = []
    trainer = PPOTrainerSeparateAsync.__new__(PPOTrainerSeparateAsync)
    _pending_validation_state(trainer, events)
    trainer.current_mode = HybridEngineMode.ROLLOUT
    trainer.timing_raw = {}
    trainer.hybrid_rollout_config = SimpleNamespace(enable_switch=True)
    trainer.config = SimpleNamespace(data=SimpleNamespace(train_batch_size=64))
    trainer.parameter_sync_step = 4
    trainer._switch_threshold_ratio = 0.25
    trainer.replay_buffer = MagicMock()

    trainer.on_step_begin()

    assert trainer.current_mode == HybridEngineMode.ROLLOUT
    assert trainer._step_threshold == 16
    trainer.replay_buffer.get_sampleable_count.assert_not_called()
