# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chat rendering by Affordance (prototype).

``--tokenizer-mode affordance`` keeps the plain HF tokenizer for token operations and replaces
the template stage: a chat request is rendered straight to token ids by an Affordance template
view (a TSX port of the model's chat template that produces the same ids), instead of a Jinja
render followed by a tokenizer pass.

The view is named by the environment for now:

- ``VLLM_AFFORDANCE_VIEW``: the template view, a ``.tsx`` file
- ``VLLM_AFFORDANCE_ENTRY``: its component (for example ``DeepSeekV41``)
- ``VLLM_AFFORDANCE_TOKENIZER``: the model's ``tokenizer.json`` (default: beside the model)

Requests the view does not cover (media, a ``chat_template`` given in the request) are not
handled yet: they raise, they are not rendered wrongly.
"""

import json
import os
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.tokenizers.hf import HfTokenizer
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

logger = init_logger(__name__)

# What `apply_chat_template` takes that is not an argument of the template itself.
_NOT_TEMPLATE_ARGUMENTS = {"chat_template", "return_dict", "tokenize", "tools", "documents"}


class AffordanceRenderer(BaseRenderer[HfTokenizer]):
    def __init__(self, config: VllmConfig, tokenizer: HfTokenizer | None) -> None:
        super().__init__(config, tokenizer)

        import affordance_render  # the binding; imported here so other modes do not need it

        self._render_ids = affordance_render.render
        self._view = os.environ["VLLM_AFFORDANCE_VIEW"]
        self._entry = os.environ["VLLM_AFFORDANCE_ENTRY"]
        self._tokenizer_json = os.environ.get("VLLM_AFFORDANCE_TOKENIZER") or os.path.join(
            str(config.model_config.tokenizer), "tokenizer.json"
        )
        special = {}
        if tokenizer is not None:
            for name in ("bos_token", "eos_token", "pad_token", "unk_token"):
                value = getattr(tokenizer, name, None)
                if isinstance(value, str):
                    special[name] = value
        self._special = json.dumps(special)
        self._render_async = make_async(self._render, executor=self._executor)
        logger.info("Chat rendering by Affordance: %s (%s)", self._view, self._entry)

    def _render(self, conversation: list[ConversationMessage], **kwargs: Any) -> list[int]:
        if kwargs.get("chat_template"):
            raise ValueError("a request-level chat_template is not supported with --tokenizer-mode affordance")
        case = {
            "messages": conversation,
            "tools": kwargs.get("tools"),
            "kwargs": {k: v for k, v in kwargs.items() if k not in _NOT_TEMPLATE_ARGUMENTS and v is not None},
        }
        # The GIL is released inside; a view and its caches are kept per renderer thread.
        return self._render_ids(self._view, self._entry, self._tokenizer_json, self._special, json.dumps(case))

    def _build_prompt(self, ids: list[int], mm_data, mm_uuids) -> DictPrompt:
        if mm_data:
            raise ValueError("media inputs are not supported with --tokenizer-mode affordance yet")
        return parse_dec_only_prompt(ids)

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        ids = self._render(conversation, **params.get_apply_chat_template_kwargs())
        return conversation, self._build_prompt(ids, mm_data, mm_uuids)

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        ids = await self._render_async(conversation, **params.get_apply_chat_template_kwargs())
        return conversation, self._build_prompt(ids, mm_data, mm_uuids)
