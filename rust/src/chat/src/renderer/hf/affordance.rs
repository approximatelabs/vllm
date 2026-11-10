// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Chat rendering by Affordance (prototype).
//!
//! A chat request is rendered straight to token ids by an Affordance template view: a TSX port
//! of the model's chat template that produces the ids the template plus the tokenizer would.
//! There is no Jinja render and no tokenizer pass, and a conversation that grows by a turn
//! reuses the previous render.
//!
//! The request is converted exactly as for the Hugging Face renderer (the same message, tool
//! and keyword-argument shapes), so the two renderers see the same inputs.
//!
//! Configuration, by environment for now:
//! - `VLLM_AFFORDANCE_VIEW`: the template view, a `.tsx` file
//! - `VLLM_AFFORDANCE_ENTRY`: its component, e.g. `DeepSeekV41`
//! - `VLLM_AFFORDANCE_TOKENIZER`: the model's `tokenizer.json`
//!
//! Not handled yet, and refused instead of rendered wrongly: media content, a per-request
//! `chat_template`, `continue_final_message`, `documents`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use affordance::emit::{Encoder, RunCache};
use affordance::{View, state_from_json};
use serde_json::{Value as JsonValue, json};
use vllm_text::Prompt;

use super::format::ChatTemplateContentFormat;
use super::{effective_template_kwargs, to_template_messages, to_template_tools};
use crate::error::{Error, Result};
use crate::renderer::{ChatRenderer, RenderedPrompt};
use crate::request::ChatRequest;

/// A view, its tokenizer and its render cache. The interpreter's values are not shared between
/// threads, so each rendering thread loads its own.
struct Loaded {
    enc: Encoder,
    view: View,
    cache: RunCache,
}

thread_local! {
    static LOADED: RefCell<HashMap<PathBuf, Loaded>> = RefCell::new(HashMap::new());
}

/// Renders chat requests to token ids with an Affordance template view.
pub struct AffordanceChatRenderer {
    view: PathBuf,
    entry: String,
    tokenizer: PathBuf,
    /// The tokenizer's special tokens as the template names them (`bos_token`, ...).
    special: JsonValue,
    default_template_kwargs: HashMap<String, JsonValue>,
}

fn setting(name: &str) -> Result<String> {
    std::env::var(name)
        .map_err(|_| Error::ChatTemplate(format!("the affordance renderer needs {name} to be set")))
}

/// `bos_token` and the like from `tokenizer_config.json`: a string, or `{"content": ...}`.
fn special_tokens(tokenizer_config: Option<&Path>) -> JsonValue {
    let mut out = serde_json::Map::new();
    let config: JsonValue = tokenizer_config
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(JsonValue::Null);
    for name in ["bos_token", "eos_token", "pad_token", "unk_token"] {
        let token = match &config[name] {
            JsonValue::String(text) => Some(text.clone()),
            JsonValue::Object(fields) => fields.get("content").and_then(|c| c.as_str()).map(str::to_owned),
            _ => None,
        };
        if let Some(token) = token {
            out.insert(name.to_string(), JsonValue::String(token));
        }
    }
    JsonValue::Object(out)
}

impl AffordanceChatRenderer {
    /// A renderer for the view the environment names, checked by loading it once.
    pub fn from_env(
        tokenizer_config: Option<&Path>,
        default_template_kwargs: HashMap<String, JsonValue>,
    ) -> Result<Self> {
        let renderer = Self {
            view: setting("VLLM_AFFORDANCE_VIEW")?.into(),
            entry: setting("VLLM_AFFORDANCE_ENTRY")?,
            tokenizer: setting("VLLM_AFFORDANCE_TOKENIZER")?.into(),
            special: special_tokens(tokenizer_config),
            default_template_kwargs,
        };
        renderer.load()?;
        Ok(renderer)
    }

    /// A renderer for a given view (tests, benchmarks).
    pub fn new(view: PathBuf, entry: String, tokenizer: PathBuf, special: JsonValue) -> Result<Self> {
        let renderer = Self { view, entry, tokenizer, special, default_template_kwargs: HashMap::new() };
        renderer.load()?;
        Ok(renderer)
    }

    fn load(&self) -> Result<Loaded> {
        let tokenizer = self.tokenizer.to_string_lossy();
        let enc = Encoder::from_file(&tokenizer).map_err(Error::ChatTemplate)?;
        let view = View::load(&self.view, &self.entry, &enc).map_err(Error::ChatTemplate)?;
        Ok(Loaded { enc, view, cache: RunCache::new(true) })
    }
}

impl ChatRenderer for AffordanceChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        if request.chat_options.chat_template.is_some() {
            return Err(Error::UntrustedRequestChatTemplate);
        }
        if request.chat_options.continue_final_message() {
            return Err(Error::ChatTemplate(
                "the affordance renderer does not support continue_final_message yet".to_string(),
            ));
        }
        if request.documents.is_some() {
            return Err(Error::ChatTemplate(
                "the affordance renderer does not support documents yet".to_string(),
            ));
        }
        // No multimodal info: a media part is refused by the conversion.
        let messages = to_template_messages(&request.messages, ChatTemplateContentFormat::String, None)?;
        let tools = (!request.tools().is_empty()).then(|| to_template_tools(request.tools()));
        let effective_template_kwargs = effective_template_kwargs(&self.default_template_kwargs, request);
        let mut kwargs = serde_json::Map::new();
        for (name, value) in &effective_template_kwargs {
            kwargs.insert(name.clone(), value.clone());
        }
        kwargs.insert(
            "add_generation_prompt".to_string(),
            json!(request.chat_options.add_generation_prompt()),
        );
        let case = json!({"messages": messages, "tools": tools, "kwargs": kwargs});

        let ids = LOADED.with(|loaded| -> Result<Vec<u32>> {
            let mut loaded = loaded.borrow_mut();
            if !loaded.contains_key(&self.view) {
                loaded.insert(self.view.clone(), self.load()?);
            }
            let l = loaded.get_mut(&self.view).expect("loaded above");
            let state = state_from_json(&case, &self.special);
            let now = chrono::Utc::now().naive_utc();
            let rendered = l.view.render(&l.enc, state, &mut l.cache, now).map_err(|error| {
                // What the official template raises, the view raises: a request error.
                match error.strip_prefix("raise: ") {
                    Some(message) => Error::ChatTemplateThrown { message: message.to_owned() },
                    None => Error::ChatTemplate(error),
                }
            })?;
            Ok(rendered.ids)
        })?;

        Ok(RenderedPrompt {
            prompt: Prompt::TokenIds(ids),
            media_order: None,
            effective_template_kwargs,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::Instant;

    use serde_json::Value as JsonValue;
    use vllm_text::backend::hf::load_tokenizer_config;
    use vllm_tokenizer::{HuggingFaceTokenizer, Tokenizer as _};

    use super::AffordanceChatRenderer;
    use crate::ChatRenderer;
    use crate::renderer::hf::{ChatTemplateContentFormatOption, HfChatRenderer, load_chat_template};
    use crate::request::{ChatMessage, ChatRequest, ResolvedToolContext};

    fn request(messages: Vec<ChatMessage>) -> ChatRequest {
        let tool_context = ResolvedToolContext::new(&messages, Vec::new(), None, true).unwrap();
        ChatRequest { messages, tool_context, request_id: "race".to_string(), ..ChatRequest::for_test() }
    }

    /// The race: this frontend's own path (minijinja, then the tokenizer) against the Affordance
    /// view, on a conversation that grows by a turn a request, as an agent's does. The ids must be
    /// equal at every step. Needs an affordance checkout with its reference models:
    ///
    ///   AFFORDANCE_ROOT=/path/to/affordance AFFORDANCE_MODEL=zai-org__GLM-5.3 \
    ///     cargo test -p vllm-chat --release affordance_race -- --ignored --nocapture
    #[test]
    #[ignore = "needs AFFORDANCE_ROOT and AFFORDANCE_MODEL"]
    fn affordance_race() {
        let root = PathBuf::from(std::env::var("AFFORDANCE_ROOT").unwrap());
        let model = std::env::var("AFFORDANCE_MODEL").unwrap();
        let index: JsonValue =
            serde_json::from_str(&std::fs::read_to_string(root.join("views/index.json")).unwrap()).unwrap();
        let view = root.join("views").join(index[&model]["view"].as_str().unwrap());
        let entry = index[&model]["entry"].as_str().unwrap().to_string();
        let dir = root.join("ref/models").join(&model);
        let tokenizer_json = root.join(format!("ref/tokenizers/{model}.json"));
        let special: JsonValue = serde_json::from_str(
            &std::fs::read_to_string(root.join(format!("ref/tokenizers/{model}.special.json"))).unwrap(),
        )
        .unwrap();

        let config = load_tokenizer_config(Some(&dir.join("tokenizer_config.json"))).unwrap();
        let template = load_chat_template(&dir.join("chat_template.jinja")).unwrap().or(config.chat_template);
        let jinja = HfChatRenderer::new(template, HashMap::new(), ChatTemplateContentFormatOption::Auto)
            .unwrap()
            .with_special_tokens(Some(config.special_tokens));
        let tokenizer = HuggingFaceTokenizer::new(&dir.join("tokenizer.json")).unwrap();
        let ours = AffordanceChatRenderer::new(view, entry, tokenizer_json, special).unwrap();

        let text = "The quick brown fox jumps over the lazy dog. ".repeat(40);
        let mut turns = vec![ChatMessage::system("You are a careful coding assistant.")];
        let (mut jinja_render, mut jinja_tokenize, mut affordance) = (0u128, 0u128, 0u128);
        let (mut equal, mut different, mut last) = (0, 0, 0);
        let steps = 200;
        for i in 0..steps {
            turns.push(ChatMessage::user(format!("Step {i}: {text}")));
            turns.push(ChatMessage::assistant_text(format!("Done with step {i}. {text}")));
            let mut messages = turns.clone();
            messages.push(ChatMessage::user("continue".to_string()));
            let request = request(messages);

            let t = Instant::now();
            let prompt = jinja.render(&request).unwrap().prompt.into_text().unwrap();
            jinja_render += t.elapsed().as_nanos();
            let t = Instant::now();
            let want = tokenizer.encode(&prompt, false).unwrap();
            jinja_tokenize += t.elapsed().as_nanos();

            let t = Instant::now();
            let got = ours.render(&request).unwrap().prompt.into_token_ids().unwrap();
            affordance += t.elapsed().as_nanos();

            if got == want { equal += 1 } else { different += 1 }
            last = want.len();
        }
        let ms = |ns: u128| ns as f64 / 1e6;
        println!(
            "{model}: {steps} requests growing to {last} tokens; ids equal on {equal}, different on {different}\n  \
             minijinja {:.0} ms + tokenize {:.0} ms = {:.0} ms   affordance {:.0} ms   {:.1}x",
            ms(jinja_render),
            ms(jinja_tokenize),
            ms(jinja_render + jinja_tokenize),
            ms(affordance),
            (jinja_render + jinja_tokenize) as f64 / affordance as f64,
        );
        assert_eq!(different, 0, "the two renderers must produce the same ids");
    }
}
