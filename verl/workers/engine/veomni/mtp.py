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

import torch


def _row_values(tensor: torch.Tensor, row: int) -> torch.Tensor:
    """Return one row from a padded or jagged one-dimensional mask."""
    if tensor.is_nested:
        values = tensor.values()
        offsets = tensor.offsets().to(torch.long)
        return values[int(offsets[row]) : int(offsets[row + 1])]
    return tensor[row]


def _build_sample_labels(
    sample_ids: torch.Tensor,
    response_mask_row: torch.Tensor,
    attention_mask_row: torch.Tensor | None,
    input_is_packed: bool,
    response_mask_is_full_length: bool,
    num_depths: int,
    ignore_index: int,
) -> list[torch.Tensor]:
    sample_length = sample_ids.numel()
    sequence_length = sample_length

    if response_mask_is_full_length:
        response_start = 0
    else:
        response_width = response_mask_row.numel()
        if attention_mask_row is not None:
            sequence_length = int(attention_mask_row.sum().item())
        elif not input_is_packed:
            raise ValueError(
                "Padded input_ids with a response-width response_mask require attention_mask "
                "to locate the response suffix."
            )
        if response_width > sequence_length:
            raise ValueError(
                f"response_mask width ({response_width}) exceeds sequence length ({sequence_length})."
            )
        response_start = sequence_length - response_width

    depth_rows = []
    for depth in range(num_depths):
        labels = torch.full(
            (sample_length,),
            fill_value=ignore_index,
            dtype=torch.long,
            device=sample_ids.device,
        )
        for position in range(sequence_length):
            target_index = position + depth + 2
            if target_index >= sequence_length:
                continue
            if not response_mask_is_full_length:
                response_index = target_index - response_start
                if response_index < 0 or response_index >= response_mask_row.numel():
                    continue
                if not bool(response_mask_row[response_index].item()):
                    continue
            elif not bool(response_mask_row[target_index].item()):
                continue
            if attention_mask_row is not None and not bool(attention_mask_row[target_index].item()):
                continue
            labels[position] = sample_ids[target_index].to(torch.long)
        depth_rows.append(labels)
    return depth_rows


def build_mtp_labels(
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    num_depths: int,
    ignore_index: int = -100,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build future-token labels for Qwen3.5-style MTP training.

    For MTP depth ``d``, position ``i`` predicts ``input_ids[i + d + 2]``. A target is
    valid only when it lies inside the same sample and its response mask is one. Packed
    batches are concatenated per sample, so the result has shape ``(1, D, total_nnz)``;
    padded BSHD batches keep shape ``(B, D, S)``.
    """
    if not isinstance(num_depths, int) or num_depths <= 0:
        raise ValueError(f"num_depths must be a positive integer, got {num_depths!r}.")
    if input_ids.is_nested:
        if input_ids.dim() != 2:
            raise ValueError(f"Packed input_ids must have rank 2, got shape {tuple(input_ids.shape)}.")
    elif input_ids.dim() != 2:
        raise ValueError(f"input_ids must have rank 2, got shape {tuple(input_ids.shape)}.")
    if response_mask.dim() != 2:
        raise ValueError(f"response_mask must have rank 2, got shape {tuple(response_mask.shape)}.")

    batch_size = input_ids.shape[0]
    if response_mask.shape[0] != batch_size:
        raise ValueError(
            f"response_mask batch size ({response_mask.shape[0]}) does not match input_ids ({batch_size})."
        )
    if attention_mask is not None:
        if attention_mask.dim() != 2 or attention_mask.shape[0] != batch_size:
            raise ValueError(
                f"attention_mask must have rank 2 and batch size {batch_size}, "
                f"got shape {tuple(attention_mask.shape)}."
            )

    all_depth_rows: list[list[torch.Tensor]] = [[] for _ in range(num_depths)]
    for row in range(batch_size):
        if input_ids.is_nested:
            input_values = input_ids.values()
            offsets = input_ids.offsets().to(torch.long)
            sample_ids = input_values[int(offsets[row]) : int(offsets[row + 1])]
        else:
            sample_ids = input_ids[row]

        response_mask_row = _row_values(response_mask, row)
        attention_mask_row = None if attention_mask is None else _row_values(attention_mask, row)

        if input_ids.is_nested:
            response_mask_is_full_length = response_mask_row.numel() == sample_ids.numel()
        else:
            response_mask_is_full_length = not response_mask.is_nested and response_mask.shape[1] == input_ids.shape[1]

        sample_depth_rows = _build_sample_labels(
            sample_ids=sample_ids,
            response_mask_row=response_mask_row,
            attention_mask_row=attention_mask_row,
            input_is_packed=input_ids.is_nested,
            response_mask_is_full_length=response_mask_is_full_length,
            num_depths=num_depths,
            ignore_index=ignore_index,
        )
        for depth in range(num_depths):
            all_depth_rows[depth].append(sample_depth_rows[depth])

    if input_ids.is_nested:
        depth_rows = [torch.cat(rows, dim=0) for rows in all_depth_rows]
        return torch.stack(depth_rows, dim=0).unsqueeze(0)

    return torch.stack([torch.stack(rows, dim=0) for rows in all_depth_rows], dim=1)


def count_mtp_targets(
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    num_depths: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Count valid MTP targets without materializing label tensors.

    This is used before micro-batching to obtain the global normalization count while
    :func:`build_mtp_labels` remains responsible for the actual training labels.
    """
    if not isinstance(num_depths, int) or num_depths <= 0:
        raise ValueError(f"num_depths must be a positive integer, got {num_depths!r}.")
    if input_ids.is_nested:
        if input_ids.dim() != 2:
            raise ValueError(f"Packed input_ids must have rank 2, got shape {tuple(input_ids.shape)}.")
    elif input_ids.dim() != 2:
        raise ValueError(f"input_ids must have rank 2, got shape {tuple(input_ids.shape)}.")
    if response_mask.dim() != 2:
        raise ValueError(f"response_mask must have rank 2, got shape {tuple(response_mask.shape)}.")

    batch_size = input_ids.shape[0]
    if response_mask.shape[0] != batch_size:
        raise ValueError(
            f"response_mask batch size ({response_mask.shape[0]}) does not match input_ids ({batch_size})."
        )
    if attention_mask is not None and (attention_mask.dim() != 2 or attention_mask.shape[0] != batch_size):
        raise ValueError(
            f"attention_mask must have rank 2 and batch size {batch_size}, "
            f"got shape {tuple(attention_mask.shape)}."
        )

    total = 0
    input_offsets = input_ids.offsets().to(torch.long) if input_ids.is_nested else None

    for row in range(batch_size):
        if input_ids.is_nested:
            sample_length = int(input_offsets[row + 1] - input_offsets[row])
            sequence_length = sample_length
        else:
            sample_length = input_ids.shape[1]
            sequence_length = sample_length

        response_mask_row = _row_values(response_mask, row)
        attention_mask_row = None if attention_mask is None else _row_values(attention_mask, row)
        if input_ids.is_nested:
            response_mask_is_full_length = response_mask_row.numel() == sample_length
        else:
            response_mask_is_full_length = not response_mask.is_nested and response_mask.shape[1] == input_ids.shape[1]

        if response_mask_is_full_length:
            response_start = 0
            target_mask = response_mask_row.to(bool)
        else:
            response_width = response_mask_row.numel()
            if attention_mask_row is not None:
                sequence_length = int(attention_mask_row.sum().item())
            elif not input_ids.is_nested:
                raise ValueError(
                    "Padded input_ids with a response-width response_mask require attention_mask "
                    "to locate the response suffix."
                )
            if response_width > sequence_length:
                raise ValueError(
                    f"response_mask width ({response_width}) exceeds sequence length ({sequence_length})."
                )
            response_start = sequence_length - response_width
            target_mask = response_mask_row.to(bool)

        target_positions = torch.arange(
            response_start,
            sequence_length,
            device=response_mask_row.device,
            dtype=torch.long,
        )
        if target_positions.numel() != target_mask.numel():
            raise ValueError(
                f"response_mask width ({target_mask.numel()}) does not match target positions "
                f"({target_positions.numel()})."
            )
        if attention_mask_row is not None:
            target_mask = target_mask & attention_mask_row[target_positions].to(bool)

        # For target position t, valid depths d satisfy i = t - d - 2 >= 0 and
        # d < num_depths, so each response target contributes min(D, t - 1).
        depth_counts = torch.clamp(target_positions - 1, min=0)
        depth_counts = torch.minimum(depth_counts, depth_counts.new_full((), num_depths))
        total += int((target_mask.to(depth_counts.dtype) * depth_counts).sum().item())

    return torch.tensor(total, dtype=torch.long, device=input_ids.device if input_ids.is_nested else input_ids.device)
