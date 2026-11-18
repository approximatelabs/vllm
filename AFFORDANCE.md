# vLLM with Affordance building its prompts

This is [vLLM](https://github.com/vllm-project/vllm) 0.33.0, a model server, with one change.
When a chat request arrives, vLLM turns its messages into the tokens the model reads: it fills in
the model's chat template, then splits the text into tokens. On this branch
[Affordance](https://github.com/approximatelabs/affordance) builds those tokens instead, from a
template app written in TSX. It keeps the tokens of the turns it has already built, so a request
in a long conversation only builds its new turn.

On 2,880 test requests, including ones with tools, vLLM produced the same tokens with Affordance as
without it. Timings are on
[affordance.rs/showcase/first-token](https://affordance.rs/showcase/first-token).

## Running it

The change is a renderer in vLLM's Rust frontend
(`rust/src/chat/src/renderer/hf/affordance.rs`). Build the frontend as upstream does, and choose
the renderer with `--tokenizer-mode affordance`. Name an Affordance template app for the model
with three environment variables. The apps are in the Affordance repository, listed by model in
`views/index.json`. For Qwen3.6-35B-A3B:

```sh
git clone https://github.com/approximatelabs/affordance
export VLLM_AFFORDANCE_VIEW=$PWD/affordance/views/third_party/qwen/Qwen3.6-35B-A3B.tsx
export VLLM_AFFORDANCE_ENTRY=Main
export VLLM_AFFORDANCE_TOKENIZER=/path/to/Qwen3.6-35B-A3B/tokenizer.json
vllm-rs serve Qwen/Qwen3.6-35B-A3B --tokenizer-mode affordance
```

What the app does not render (pictures and other media, a per-request `chat_template`,
`continue_final_message`, `documents`) goes to vLLM's Hugging Face renderer, as without this
branch.

This branch is the code behind the measurements above, kept on vLLM 0.33.0. It does not follow
vLLM's releases.
