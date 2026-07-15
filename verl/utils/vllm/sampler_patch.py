# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Patch vLLM V1 Sampler to avoid OOM in ``log_softmax`` when ``logprobs=0``.

When ``calculate_log_probs=True`` (verl's default for async / bypass mode), vLLM
receives ``SamplingParams.logprobs=0``, meaning only the sampled token's logprob
is needed.  However, the upstream ``Sampler`` still materialises a full
``[N, vocab_size]`` float32 log-softmax matrix and then ``gather``s a single
value per row.  For large vocab models (e.g. Qwen2.5 with 152k tokens) this
causes OOM during the sample step.

This patch intercepts ``Sampler.forward`` for the ``num_logprobs == 0`` case and
computes the sampled-token logprob directly via ``gather`` + ``logsumexp``,
avoiding the ``[N, vocab_size]`` float32 allocation entirely.  Peak memory for
the logprobs path drops from ``O(N * vocab * 4)`` to ``O(N)``.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_patched = False


def patch_vllm_sampler_for_logprobs_0():
    """Monkey-patch ``vllm.v1.sample.sampler.Sampler``.

    Idempotent: calling more than once is a no-op.
    """
    global _patched
    if _patched:
        return

    from vllm.v1.outputs import LogprobsTensors, SamplerOutput
    from vllm.v1.sample.sampler import Sampler

    _original_forward = Sampler.forward

    def patched_forward(
        self,
        logits: torch.Tensor,
        sampling_metadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override=None,
    ) -> SamplerOutput:
        num_logprobs = sampling_metadata.max_num_logprobs

        # Fast path: only the sampled token's logprob is requested.
        # This is the common case for verl with calculate_log_probs=True.
        if num_logprobs is not None and num_logprobs == 0:
            return _forward_logprobs_0(
                self, logits, sampling_metadata, predict_bonus_token
            )

        # Fall back to original implementation for other cases.
        return _original_forward(
            self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
        )

    def _forward_logprobs_0(
        self,
        logits: torch.Tensor,
        sampling_metadata,
        predict_bonus_token: bool,
    ) -> SamplerOutput:
        # Convert to float32 (same as original forward).
        logits = logits.to(torch.float32)
        logits = self.apply_logits_processors(logits, sampling_metadata, predict_bonus_token)

        # Temporarily disable logprobs computation inside sample().
        # Setting logprobs_mode to a value outside ("processed_logits",
        # "processed_logprobs") makes sample() return None for processed_logprobs,
        # skipping the expensive [N, vocab] log_softmax.
        original_mode = self.logprobs_mode
        original_sampler_mode = self.topk_topp_sampler.logprobs_mode
        self.logprobs_mode = "raw_logprobs"
        self.topk_topp_sampler.logprobs_mode = "raw_logprobs"

        try:
            sampled, _ = self.sample(logits, sampling_metadata)
        finally:
            self.logprobs_mode = original_mode
            self.topk_topp_sampler.logprobs_mode = original_sampler_mode

        sampled = sampled.long()

        # Compute sampled-token logprob without materialising the full
        # [N, vocab] log-softmax matrix.
        #
        #   log_softmax(x)[i, t] = x[i, t] - logsumexp(x[i, :])
        #
        # Both gather and logsumexp produce [N, 1] / [N] tensors, so peak
        # memory for this step is O(N) instead of O(N * vocab).
        token_ids = sampled.unsqueeze(-1)  # [N, 1]
        token_logits = logits.gather(-1, token_ids)  # [N, 1]
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)  # [N, 1]
        token_logprobs = token_logits - lse  # [N, 1]

        # Compute token ranks: how many tokens have logit >= sampled token's logit.
        # log_softmax is monotonic, so comparing logits is equivalent to comparing
        # logprobs.  Uses >= to match vLLM's batched_count_greater_than semantics.
        # Operates on the already-allocated float32 logits -- no new [N, vocab] tensor.
        token_ranks = (logits >= token_logits).sum(dim=-1)  # [N]

        logprobs_tensors = LogprobsTensors(
            token_ids.to(torch.int32),  # indices: [N, 1]
            token_logprobs,  # logprobs: [N, 1]
            token_ranks,  # ranks: [N]
        )

        sampled = sampled.to(torch.int32)
        return SamplerOutput(
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )

    Sampler.forward = patched_forward
    _patched = True
    logger.info(
        "Patched vLLM Sampler.forward for logprobs=0 fast path "
        "(avoids full [N, vocab] log_softmax materialisation)"
    )
