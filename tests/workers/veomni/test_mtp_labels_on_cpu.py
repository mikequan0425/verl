# Copyright 2026 Bytedance Ltd. and/or its affiliates.
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

import importlib.util
from pathlib import Path

import pytest
import torch

_MTP_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "verl/workers/engine/veomni/mtp.py"
)
_MTP_SPEC = importlib.util.spec_from_file_location("_verl_veomni_mtp", _MTP_MODULE_PATH)
_MTP = importlib.util.module_from_spec(_MTP_SPEC)
_MTP_SPEC.loader.exec_module(_MTP)

build_mtp_labels = _MTP.build_mtp_labels
count_mtp_targets = _MTP.count_mtp_targets


def test_packed_mtp_labels_do_not_roll_across_samples():
    input_ids = torch.nested.nested_tensor_from_jagged(
        torch.tensor([10, 11, 20, 21, 22, 30, 31, 40]),
        offsets=torch.tensor([0, 5, 8]),
    )
    response_mask = torch.nested.nested_tensor_from_jagged(
        torch.tensor([0, 0, 1, 1, 1, 0, 0, 1], dtype=torch.bool),
        offsets=torch.tensor([0, 5, 8]),
    )

    labels = build_mtp_labels(input_ids, response_mask, num_depths=2)

    assert labels.shape == (1, 2, 8)
    torch.testing.assert_close(
        labels[0, 0],
        torch.tensor([20, 21, 22, -100, -100, 40, -100, -100]),
    )
    torch.testing.assert_close(
        labels[0, 1],
        torch.tensor([21, 22, -100, -100, -100, -100, -100, -100]),
    )


def test_padded_bshd_mtp_labels_keep_layout_and_ignore_static_padding():
    input_ids = torch.tensor(
        [
            [10, 11, 20, 21, 22],
            [30, 31, 40, 99, 99],
        ]
    )
    response_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )

    labels = build_mtp_labels(
        input_ids,
        response_mask,
        num_depths=2,
        attention_mask=attention_mask,
    )

    assert labels.shape == (2, 2, 5)
    torch.testing.assert_close(labels[:, 0], torch.tensor([[20, 21, 22, -100, -100], [40, -100, -100, -100, -100]]))
    torch.testing.assert_close(labels[:, 1], torch.tensor([[21, 22, -100, -100, -100], [-100] * 5]))


def test_padded_response_width_mask_maps_to_sequence_suffix():
    input_ids = torch.tensor([[10, 11, 20, 21, 22]])
    response_mask = torch.tensor([[0, 0, 1]], dtype=torch.bool)
    attention_mask = torch.ones(1, 5, dtype=torch.bool)

    labels = build_mtp_labels(
        input_ids,
        response_mask,
        num_depths=1,
        attention_mask=attention_mask,
    )

    torch.testing.assert_close(labels[0, 0], torch.tensor([-100, -100, 22, -100, -100]))


def test_packed_response_width_mask_does_not_require_attention_mask():
    input_ids = torch.nested.nested_tensor_from_jagged(
        torch.tensor([10, 11, 20]),
        offsets=torch.tensor([0, 3]),
    )
    response_mask = torch.nested.nested_tensor_from_jagged(
        torch.tensor([1], dtype=torch.bool),
        offsets=torch.tensor([0, 1]),
    )

    labels = build_mtp_labels(input_ids, response_mask, num_depths=1)

    assert labels.shape == (1, 1, 3)
    torch.testing.assert_close(labels[0, 0], torch.tensor([20, -100, -100]))


def test_padded_bshd_mtp_labels_use_batch_first_depth_layout():
    input_ids = torch.tensor([[10, 11, 20, 21, 22]])
    response_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)

    labels = build_mtp_labels(input_ids, response_mask, num_depths=1)

    assert labels.shape == (1, 1, 5)


def test_count_mtp_targets_matches_label_count():
    input_ids = torch.nested.nested_tensor_from_jagged(
        torch.tensor([10, 11, 20, 21, 22, 30, 31, 40]),
        offsets=torch.tensor([0, 5, 8]),
    )
    response_mask = torch.nested.nested_tensor_from_jagged(
        torch.tensor([0, 0, 1, 1, 1, 0, 0, 1], dtype=torch.bool),
        offsets=torch.tensor([0, 5, 8]),
    )

    labels = build_mtp_labels(input_ids, response_mask, num_depths=2)

    assert count_mtp_targets(input_ids, response_mask, num_depths=2) == (labels != -100).sum()


def test_count_mtp_targets_uses_attention_mask_for_padded_bshd():
    input_ids = torch.tensor(
        [
            [10, 11, 20, 21, 22],
            [30, 31, 40, 99, 99],
        ]
    )
    response_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    labels = build_mtp_labels(
        input_ids,
        response_mask,
        num_depths=2,
        attention_mask=attention_mask,
    )

    assert count_mtp_targets(input_ids, response_mask, num_depths=2, attention_mask=attention_mask) == (
        labels != -100
    ).sum()


def test_mtp_labels_validate_inputs():
    input_ids = torch.tensor([[1, 2, 3]])
    response_mask = torch.tensor([[0, 0, 1]], dtype=torch.bool)

    with pytest.raises(ValueError, match="positive integer"):
        build_mtp_labels(input_ids, response_mask, num_depths=0)
    with pytest.raises(ValueError, match="rank 2"):
        build_mtp_labels(input_ids, response_mask[0], num_depths=1)
    with pytest.raises(ValueError, match="require attention_mask"):
        build_mtp_labels(input_ids, response_mask[:, :2], num_depths=1)
