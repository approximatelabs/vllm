// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Chat rendering by Affordance (prototype).
//!
//! A chat request is rendered straight to token ids by an Affordance template view: a TSX port
//! of the model's chat template that produces the ids the template plus the tokenizer would.
//! There is no Jinja render and no tokenizer pass, and a conversation that grows by a turn
//! reuses the previous render.
//!
//! The renderer owns its rendering threads. A conversation (keyed by its first two messages)
//! always goes to the same thread, which keeps the last few dozen conversations it rendered.
//!
//! The request is converted exactly as for the Hugging Face renderer (the same message, tool
//! and keyword-argument shapes), so the two renderers see the same inputs.
//!
//! Configuration, by environment for now:
//! - `VLLM_AFFORDANCE_VIEW`: the template view, a `.tsx` file
//! - `VLLM_AFFORDANCE_ENTRY`: its component, e.g. `DeepSeekV41`
//! - `VLLM_AFFORDANCE_TOKENIZER`: the model's `tokenizer.json`
//!
//! What the view does not render (media content, a per-request `chat_template`,
//! `continue_final_message`, `documents`) goes to the Hugging Face renderer, as without this
//! renderer; with no fallback given it is refused, never rendered wrongly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::{Value as JsonValue, json};
use vllm_text::Prompt;

use super::format::ChatTemplateContentFormat;
use super::{HfChatRenderer, effective_template_kwargs, to_template_message, to_template_tools};
use crate::error::{Error, Result};
use crate::renderer::{ChatRenderer, RenderedPrompt};
use crate::request::{ChatMessage, ChatRequest};

/// Renders chat requests to token ids with an Affordance template view.
///
/// The rendering itself (its threads, what it keeps of each conversation) is Affordance's
/// `ChatRenderer`; this is the adapter: the request's messages, tools and template arguments
/// in the shapes the Hugging Face renderer gives a Jinja template.
pub struct AffordanceChatRenderer {
    renderer: affordance::chat_render::ChatRenderer<ChatMessage>,
    default_template_kwargs: HashMap<String, JsonValue>,
    /// The model's Jinja renderer, for requests the view does not render.
    fallback: Option<HfChatRenderer>,
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

/// A message as a Jinja chat template reads it: the Hugging Face renderer's own conversion.
/// No multimodal info is given, so a media part is refused, not rendered wrongly.
fn template_json(message: &ChatMessage) -> std::result::Result<JsonValue, String> {
    let converted =
        to_template_message(message, ChatTemplateContentFormat::String, None).map_err(|e| e.to_string())?;
    serde_json::to_value(&converted).map_err(|e| e.to_string())
}

impl AffordanceChatRenderer {
    /// A renderer for the view the environment names.
    pub fn from_env(
        tokenizer_config: Option<&Path>,
        default_template_kwargs: HashMap<String, JsonValue>,
        fallback: Option<HfChatRenderer>,
    ) -> Result<Self> {
        let mut renderer = Self::new(
            setting("VLLM_AFFORDANCE_VIEW")?.into(),
            setting("VLLM_AFFORDANCE_ENTRY")?,
            setting("VLLM_AFFORDANCE_TOKENIZER")?.into(),
            special_tokens(tokenizer_config),
        )?;
        renderer.default_template_kwargs = default_template_kwargs;
        renderer.fallback = fallback;
        Ok(renderer)
    }

    /// A renderer for a given view. A view or tokenizer that does not load is an error here.
    pub fn new(view: PathBuf, entry: String, tokenizer: PathBuf, special: JsonValue) -> Result<Self> {
        let mut options = affordance::chat_render::Options::new(view, entry, tokenizer);
        options.special = special;
        if let Some(workers) = std::env::var("VLLM_AFFORDANCE_WORKERS").ok().and_then(|n| n.parse().ok()) {
            options.workers = workers;
        }
        let renderer = affordance::chat_render::ChatRenderer::new(options, template_json).map_err(Error::ChatTemplate)?;
        Ok(Self { renderer, default_template_kwargs: HashMap::new(), fallback: None })
    }
}

impl ChatRenderer for AffordanceChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        // What the view does not render: the Jinja renderer's, if there is one.
        let unrendered = if request.chat_options.chat_template.is_some() {
            Some("a per-request chat_template")
        } else if request.chat_options.continue_final_message() {
            Some("continue_final_message")
        } else if request.documents.is_some() {
            Some("documents")
        } else {
            None
        };
        if let Some(what) = unrendered {
            return match &self.fallback {
                Some(fallback) => fallback.render(request),
                None => Err(Error::ChatTemplate(format!("the affordance renderer does not support {what}"))),
            };
        }
        let tools = (!request.tools().is_empty())
            .then(|| serde_json::to_value(to_template_tools(request.tools())))
            .transpose()
            .map_err(|e| Error::ChatTemplate(e.to_string()))?;
        let effective_template_kwargs = effective_template_kwargs(&self.default_template_kwargs, request);
        let mut kwargs = serde_json::Map::new();
        for (name, value) in &effective_template_kwargs {
            kwargs.insert(name.clone(), value.clone());
        }
        kwargs.insert(
            "add_generation_prompt".to_string(),
            json!(request.chat_options.add_generation_prompt()),
        );
        let ids = match self.renderer.render(&request.messages, tools.as_ref(), &JsonValue::Object(kwargs)) {
            Ok(ids) => ids,
            // What the official template raises, the view raises: a request error.
            Err(affordance::chat_render::RenderError::Raised(message)) => {
                return Err(Error::ChatTemplateThrown { message });
            }
            // Anything else (a message with media, which the conversion refuses) is the Jinja
            // renderer's.
            Err(affordance::chat_render::RenderError::Failed(message)) => {
                return match &self.fallback {
                    Some(fallback) => fallback.render(request),
                    None => Err(Error::ChatTemplate(message)),
                };
            }
        };
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

        // Both sides loaded before the clock starts: the view and its tokenizer are loaded on the
        // rendering thread at its first request, which is start-up, not rendering.
        let warm = request(vec![ChatMessage::user("warm up".to_string())]);
        jinja.render(&warm).unwrap();
        ours.render(&warm).unwrap();
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
    /// Many conversations at once, their requests interleaved, as a server sees them: each still
    /// costs what is new, because the renderer keeps them apart.
    #[test]
    #[ignore = "needs AFFORDANCE_ROOT and AFFORDANCE_MODEL"]
    fn affordance_race_interleaved() {
        let root = PathBuf::from(std::env::var("AFFORDANCE_ROOT").unwrap());
        let model = std::env::var("AFFORDANCE_MODEL").unwrap();
        let index: JsonValue =
            serde_json::from_str(&std::fs::read_to_string(root.join("views/index.json")).unwrap()).unwrap();
        let view = root.join("views").join(index[&model]["view"].as_str().unwrap());
        let entry = index[&model]["entry"].as_str().unwrap().to_string();
        let dir = root.join("ref/models").join(&model);
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
        let ours =
            AffordanceChatRenderer::new(view, entry, root.join(format!("ref/tokenizers/{model}.json")), special).unwrap();
        let warm = request(vec![ChatMessage::user("warm up".to_string())]);
        jinja.render(&warm).unwrap();
        ours.render(&warm).unwrap();

        let (conversations, steps) = (16, 40);
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(40);
        let mut turns: Vec<Vec<ChatMessage>> = (0..conversations)
            .map(|c| vec![ChatMessage::system("You are a careful coding assistant."), ChatMessage::user(format!("Task {c}: start."))])
            .collect();
        let (mut baseline, mut affordance, mut different, mut tokens) = (0u128, 0u128, 0, 0usize);
        for i in 0..steps {
            for conversation in turns.iter_mut() {
                conversation.push(ChatMessage::assistant_text(format!("Done with step {i}. {text}")));
                conversation.push(ChatMessage::user(format!("Step {i}: {text}")));
                let request = request(conversation.clone());
                let t = Instant::now();
                let prompt = jinja.render(&request).unwrap().prompt.into_text().unwrap();
                let want = tokenizer.encode(&prompt, false).unwrap();
                baseline += t.elapsed().as_nanos();
                let t = Instant::now();
                let got = ours.render(&request).unwrap().prompt.into_token_ids().unwrap();
                affordance += t.elapsed().as_nanos();
                different += usize::from(got != want);
                tokens = want.len();
            }
        }
        println!(
            "{model}: {conversations} conversations interleaved, {steps} requests each, to {tokens} tokens; different on {different}\n  \
             minijinja + tokenize {:.0} ms   affordance {:.0} ms   {:.1}x",
            baseline as f64 / 1e6,
            affordance as f64 / 1e6,
            baseline as f64 / affordance as f64,
        );
        assert_eq!(different, 0, "the two renderers must produce the same ids");
    }
    /// A tool-using agent: tools declared, and each step a user message, an assistant turn that
    /// calls a tool (its arguments a JSON string, as a client sends them), the tool's result and
    /// the assistant's answer. The templates' most intricate paths, and the ids must still be
    /// equal at every step.
    #[test]
    #[ignore = "needs AFFORDANCE_ROOT and AFFORDANCE_MODEL"]
    fn affordance_race_tools() {
        use crate::AssistantContentBlock;
        use crate::event::AssistantToolCall;
        use crate::request::ChatTool;
        let root = PathBuf::from(std::env::var("AFFORDANCE_ROOT").unwrap());
        let model = std::env::var("AFFORDANCE_MODEL").unwrap();
        let index: JsonValue =
            serde_json::from_str(&std::fs::read_to_string(root.join("views/index.json")).unwrap()).unwrap();
        let view = root.join("views").join(index[&model]["view"].as_str().unwrap());
        let entry = index[&model]["entry"].as_str().unwrap().to_string();
        let dir = root.join("ref/models").join(&model);
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
        let ours =
            AffordanceChatRenderer::new(view, entry, root.join(format!("ref/tokenizers/{model}.json")), special).unwrap();
        let tools = vec![
            ChatTool {
                name: "bash".to_string(),
                description: Some("Run a shell command and return its output.".to_string()),
                parameters: serde_json::json!({"type": "object", "properties": {"command": {"type": "string", "description": "The command."}, "timeout": {"type": "integer"}}, "required": ["command"]}),
                strict: None,
                defer_loading: None,
            },
            ChatTool {
                name: "read".to_string(),
                description: Some("Read a file.".to_string()),
                parameters: serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
                strict: None,
                defer_loading: None,
            },
        ];
        let with_tools = |messages: Vec<ChatMessage>| {
            let tool_context = ResolvedToolContext::new(&messages, tools.clone(), None, true).unwrap();
            ChatRequest { messages, tool_context, request_id: "race".to_string(), ..ChatRequest::for_test() }
        };
        let warm = with_tools(vec![ChatMessage::user("warm up".to_string())]);
        jinja.render(&warm).unwrap();
        ours.render(&warm).unwrap();

        let output = "src/main.rs\nsrc/lib.rs\nCargo.toml\n".repeat(30);
        let mut turns = vec![ChatMessage::system("You are a careful coding assistant.")];
        let (mut baseline, mut affordance, mut different, mut last) = (0u128, 0u128, 0, 0);
        let mut first = None;
        let steps = 60;
        for i in 0..steps {
            turns.push(ChatMessage::user(format!("Step {i}: list the files, then read one.")));
            turns.push(ChatMessage::assistant_blocks(vec![
                AssistantContentBlock::Text { text: format!("Listing for step {i}.") },
                AssistantContentBlock::ToolCall(AssistantToolCall {
                    id: format!("call_{i}_a"),
                    name: "bash".to_string(),
                    arguments: serde_json::json!({"command": format!("ls -la step{i}"), "timeout": 30}).to_string(),
                }),
            ]));
            turns.push(ChatMessage::tool_response(output.clone(), format!("call_{i}_a")));
            turns.push(ChatMessage::assistant_blocks(vec![AssistantContentBlock::ToolCall(AssistantToolCall {
                id: format!("call_{i}_b"),
                name: "read".to_string(),
                arguments: serde_json::json!({"path": format!("src/file{i}.rs")}).to_string(),
            })]));
            turns.push(ChatMessage::tool_response(format!("fn step{i}() {{}}\n").repeat(20), format!("call_{i}_b")));
            // Asked after the tool results (the model answers next), and after the answer.
            for answered in [false, true] {
                if answered {
                    turns.push(ChatMessage::assistant_text(format!("Step {i} is done.")));
                }
                let mut messages = turns.clone();
                if answered {
                    messages.push(ChatMessage::user("continue".to_string()));
                }
                let request = with_tools(messages);
                let t = Instant::now();
                let prompt = jinja.render(&request).unwrap().prompt.into_text().unwrap();
                let want = tokenizer.encode(&prompt, false).unwrap();
                baseline += t.elapsed().as_nanos();
                let t = Instant::now();
                let got = ours.render(&request).unwrap().prompt.into_token_ids().unwrap();
                affordance += t.elapsed().as_nanos();
                if got != want {
                    different += 1;
                    if first.is_none() {
                        let at = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
                        let near = |ids: &[u32]| tokenizer.decode(&ids[at.saturating_sub(12)..(at + 12).min(ids.len())], false).unwrap_or_default();
                        first = Some(format!("step {i} (answered {answered}): differs at token {at}\n    theirs: {:?}\n    ours:   {:?}", near(&want), near(&got)));
                    }
                }
                last = want.len();
            }
        }
        println!(
            "{model}: a tool-using agent, {} requests growing to {last} tokens; different on {different}\n  \
             minijinja + tokenize {:.0} ms   affordance {:.0} ms   {:.1}x",
            steps * 2,
            baseline as f64 / 1e6,
            affordance as f64 / 1e6,
            baseline as f64 / affordance as f64,
        );
        if let Some(first) = first {
            println!("  {first}");
        }
        assert_eq!(different, 0, "the two renderers must produce the same ids");
    }
}
