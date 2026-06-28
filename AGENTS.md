# xiaozhi-server-rs Agent 指南

本文件给接手此项目的 agent（包括人和 AI）使用。先通读一遍再动代码，能少走很多弯路。

> 协议参考: `/home/urie/workspace/cpp/xiaozhi-esp32/PROTOCOL.md`。
> ESP32 客户端源码: `/home/urie/workspace/cpp/xiaozhi-esp32/`。
> Go 参考实现: `/home/urie/workspace/go/xiaozhi-server-go/`。

---

## 1. 项目目标

用 Rust 实现一个 **Xiaozhi WebSocket 协议服务端**，兼容 ESP32 客户端（`xiaozhi-esp32`）：

- 提供 `/ota` 固件检查 + WebSocket 接入信息
- 提供 `/ws` WebSocket 端点处理实时音频流
- 支持完整链路：麦克风 Opus → ASR → LLM → TTS → 扬声器 Opus
- 支持**流式输出**和**打断（abort）**

当前所有外部依赖（ASR/LLM/TTS）都接了真实 provider，**默认走 mock 的部分已经全部接入真实服务**。如果你想回到 mock 模式，把 `XIAOZHI_ASR_PROVIDER` / `XIAOZHI_LLM_PROVIDER` / `XIAOZHI_TTS_PROVIDER` 留空即可。

---

## 2. 目录结构与职责

```
xiaozhi-server-rs/
├── Cargo.toml                # 依赖列表
├── shell.nix                 # Nix 开发环境（libopus、onnxruntime、rust 工具链）
├── .envrc                    # direnv 入口；导出真实 provider 的 env vars
├── models/
│   └── silero_vad.onnx       # VAD 模型（runtime 期由 silero_vad.rs 加载）
├── system_prompts/
│   └── taiwan.md             # LLM 人设提示词
└── src/
    ├── main.rs               # 入口：初始化 rustls、ONNX Runtime、tracing；启动 HTTP
    ├── config.rs             # 读取 Bind / public WS URL / token
    ├── http.rs               # axum 路由 + HTTP 中间件 + /ws upgrade
    ├── protocol.rs           # 协议层：JSON 消息 + 二进制音频帧 v1/v2/v3 编解码
    ├── session.rs            # 单个 WebSocket 会话的完整生命周期（最复杂）
    ├── text_filter.rs        # LLM 输出 → TTS 的文本过滤（去掉 markdown/emoji）
    ├── audio/
    │   ├── mod.rs
    │   ├── opus_decode.rs    # Opus → PCM i16（喂给 ASR + VAD）
    │   ├── opus_duration.rs  # 从 Opus TOC 字节解析单包时长（RFC 6716）
    │   ├── opus_silence.rs   # 一个合法 Opus 静音包，给 MockTts 用
    │   ├── ogg_opus.rs       # Ogg/Opus demuxer（火山 TTS 输出是 ogg_opus）
    │   └── silero_vad.rs     # Silero VAD ONNX 推理 + 状态机
    └── services/
        ├── mod.rs            # ServiceBundle 组装 + trait 定义
        ├── mock.rs           # MockAsr / MockLlmFactory / MockTts
        ├── openai.rs         # OpenAI 兼容 Chat Completions 流式 LLM
        ├── pi_rpc.rs         # 通过 `pi --mode rpc` 子进程调 LLM
        ├── volcengine_asr.rs # 火山引擎流式 ASR（wss + 二进制协议）
        └── volcengine.rs     # 火山引擎双向 TTS（wss + ogg_opus demux）
```

---

## 3. 关键模块契约

### 3.1 `services/mod.rs` 三个 trait

```rust
#[async_trait]
pub trait AsrService: Send + Sync + 'static {
    async fn start_stream(&self) -> Result<Box<dyn AsrStream>>;
}
pub trait AsrStream: Send + 'static {
    async fn push_pcm(&mut self, samples: &[i16]) -> Result<()>;
    async fn finish(&mut self) -> Result<String>;   // 一次性返回最终文本
    async fn abort(&mut self);
}

#[async_trait]
pub trait LlmSessionFactory: Send + Sync + 'static {
    async fn create_session(&self, meta: LlmSessionMeta) -> Result<Box<dyn LlmSession>>;
}
pub trait LlmSession: Send + 'static {
    /// 注意：chat_stream 必须是**同步**方法（不能 .await），且不能
    /// 在持有 Mutex 锁期间 await。这是 abort 能 work 的前提。详见 §6。
    fn chat_stream(&mut self, prompt: String) -> TextStream;
    async fn abort(&mut self);
    async fn shutdown(&mut self);
}

pub trait TtsService: Send + Sync + 'static {
    fn synthesize_stream(&self, input: TextStream) -> TtsStream;
}
```

类型别名：
- `TextStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>`
- `TtsStream = Pin<Box<dyn Stream<Item = Result<TtsEvent>> + Send>>`
- `TtsEvent = SentenceStart(String) | Audio(AudioFrame)`

ASR 接收的是 **PCM i16（16 kHz mono）**，不是原始 Opus。`session.rs` 在收到客户端 Opus 帧后会先 `OpusPcmDecoder::decode_to_pcm_i16`，再把 PCM 喂给 ASR **和** VAD，两边共用同一份解码结果。

### 3.2 `session.rs` 状态机

每个 WebSocket 连接 = 一个独立 session。`SessionState` 关键字段：

| 字段 | 作用 |
|------|------|
| `listening: bool` | 是否在 listen 状态；非 listen 时丢弃二进制音频 |
| `asr: Option<Box<dyn AsrStream>>` | 当前 round 的 ASR；listen.stop / VAD end 时 finish |
| `audio_decoder: Option<OpusPcmDecoder>` | 共享的 Opus 解码器，listen 期内复用 |
| `vad: Option<SileroVadStream>` | listen.start 时打开，VAD 触发前累积 pre-roll PCM |
| `pcm_ring_buffer: Option<PcmRingBuffer>` | 1s 滚动缓冲；VAD `SpeechStart` 时 drain 进 ASR |
| `llm: Arc<Mutex<Option<Box<dyn LlmSession>>>>` | **整个 WebSocket 生命周期**，不随 listen 重置 |
| `abort_notify: Arc<Notify>` | abort 时通知 pipeline 退出；pipeline 不被 cancel |

#### listen.start 的两阶段 ASR
1. `listen.start` 进来时**不**立即开 ASR；只开 VAD + PCM ring buffer。
2. 客户端发 Opus 帧 → 解码成 PCM → 一边进 ring buffer、一边进 VAD。
3. VAD 发出 `SpeechStart` → 用 ring buffer 中的 pre-roll 喂 ASR，从此每个 PCM 帧直接 push 给 ASR。
4. 触发 `SpeechEnd` 或 `listen.stop` → ASR `finish()` → pipeline。

这样做的好处：ASR 第一帧就包含 VAD 触发前的那一段音频，不会丢首字。

### 3.3 `pi_rpc.rs` 子进程模型

每个 WebSocket 连接 spawn 一个 `pi --mode rpc` 子进程。生命周期：

```
handle_websocket()
   ├─ services.llm.create_session()        # spawn 子进程
   ├─ pipeline: llm.chat_stream(prompt)    # 发 prompt
   ├─ abort_active → llm.abort()           # 发 {"type":"abort"}
   └─ ws close → llm.shutdown()            # 发 {"type":"abort"} + kill 子进程
```

#### LLM Session 契约（**最重要**，违反会导致 abort 不工作）

> `LlmSession::chat_stream` **必须**是同步方法（无 `.await`），并且实现**必须**在调用方持有 `Arc<Mutex<LlmSession>>` 期间完成调用、立即返回 `TextStream`。
>
> 这不是性能优化，是 abort 协议要求。如果在锁内 await，`abort_active` 拿不到锁就调不了 `llm.abort()`。

`pi_rpc.rs` 已经踩过这个坑的实现要点：
- `mpsc::Sender<WriterCommand>` 串行化所有对子进程 stdin 的写入
- `tokio::sync::broadcast::Sender<PiEvent>` 多消费者（reader_loop → coordinator_loop / chat_stream 各 subscribe 一份）
- `reader_loop` 区分 `assistantMessageEvent.type`：`text_delta` 进 `TextDelta`，`thinking_*` 丢弃
- `coordinator_loop` 监听 `ChildExited` / `Idle` / `shutdown_rx`，杀掉子进程
- `chat_stream` 60s idle 才报错（不是双重 continue 延长）

#### 默认 flag
当 `XIAOZHI_PI_RPC_FLAGS` 未设置时，注入安全默认值（关闭工具、技能、提示词模板、扩展、session、thinking）：
```
--no-tools --no-builtin-tools --no-skills --no-prompt-templates
--no-context-files --no-themes --no-extensions --no-session --thinking off
```

如果用户传入自己的 `XIAOZHI_PI_RPC_FLAGS`（空格分隔），**完全透传**，不做白名单校验。

---

## 4. 协议层（`protocol.rs`）

### 4.1 WebSocket JSON 消息

| 方向 | type | 字段 | 备注 |
|------|------|------|------|
| C→S | `hello` | `version` (1/2/3) | 第一次握手；协商二进制协议版本 |
| S→C | `hello` | `transport`, `session_id`, `audio_params` | format=opus, sr=24000, ch=1, frame=60ms |
| C→S | `listen` | `state: "start"\|"stop"\|"detect"`, `mode: "auto"\|"manual"\|"realtime"` | |
| S→C | `stt` | `text` | ASR 最终结果（**只在 listen.stop 时发一次**，中间不流式） |
| S→C | `llm` | `emotion`, `text` | 情绪事件（每次对话发一个） |
| S→C | `tts` | `state: "start"\|"sentence_start"\|"stop"`, 可选 `text` | `sentence_start.text` 是 best-effort 字幕，不阻塞音频 |
| C→S | `abort` | `reason` | 打断当前对话 |
| C→S | `mcp` | — | **v1 忽略**，记 debug 日志 |
| C→S | `goodbye` | — | 客户端主动断开 |

### 4.2 二进制音频帧 v1/v2/v3

服务器必须**同时**支持三种版本（设备可能缓存了旧配置）：

- **v1**：纯裸 Opus payload
- **v2**：`[ver+header=2B][type=2B][padding=4B][timestamp=4B][payload_size=4B][payload]`
- **v3**：`[ver+type=1B][padding=1B][payload_size=2B][payload]`

注意 v2/v3 的 `type` 字段必须为 0（音频）才处理，否则跳过。`encode_audio_frame` / `decode_audio_frame` 的 roundtrip 在 `protocol.rs` 测试中已覆盖。

服务端 → 客户端永远发 v1（裸 Opus），由 ESP32 client 解码。

---

## 5. HTTP 端点（`http.rs`）

| 方法 | 路径 | 行为 |
|------|------|------|
| GET/POST | `/health` | `{"status":"ok"}` |
| GET/POST | `/ota` | 返回固定 WebSocket 配置 + `firmware.version=0.0.0` + `server_time`，**不要带 `activation` 字段**（否则设备会走激活流程） |
| POST | `/ota/activate` | `200 OK`（mock） |
| GET | `/ws` | WebSocket upgrade；`Protocol-Version` header 可覆盖协商 |

认证：默认 `XIAOZHI_TOKEN=dev-token`。`Authorization` 缺失时**当前是放行的**（便于初版调试 ESP32），生产环境请改成必填。`XIAOZHI_TOKEN=""` 表示完全跳过校验。

---

## 6. 容易踩的坑（必读）

### 6.1 LLM Session 锁约定
- ❌ `let stream = self.llm.chat_stream(...).await;` — `chat_stream` 必须是同步
- ❌ `let mut guard = lock; let stream = guard.chat_stream(...); drop(guard);` 然后在 await 期间用 stream — 流是 borrow 自 guard 的，guard drop 后就悬空
- ✅ 见 `session.rs::run_pipeline`：在 `chat_stream` 调用期间一直持有锁，调用结束后立刻把流 move 出来再 await

### 6.2 abort 路径
- pipeline 任务**永远不要被 `JoinHandle::abort()`** 取消。否则会 drop 栈上的 LLM 句柄，把 pi 子进程也拖死（`kill_on_drop = true`）。
- 正确做法：`abort_notify.notify_waiters()` → pipeline 在下一个 select 点退出 → 自己发 `tts.stop` → 自己结束。
- abort 真正"打断 LLM"的路径是 `llm.abort()`（发 `{"type":"abort"}` 给子进程）。

### 6.3 ONNX Runtime / Silero VAD
- `ort` crate 用 `load-dynamic` 模式，**不会**在编译期下载 ONNX Runtime。
- `shell.nix` 通过 `ORT_DYLIB_PATH` 指向 `${pkgs.onnxruntime}/lib/libonnxruntime.so`。
- `LD_LIBRARY_PATH` 必须包含 `pkgs.onnxruntime`。
- `main.rs::init_onnx_runtime()` 在启动时就 `ort::init_from(&dylib_path).commit()`；失败直接退出，**不要**等 listen.start 才报错。
- 默认开启 VAD；想关掉设 `XIAOZHI_VAD_PROVIDER=none`。
- 模型默认路径 `models/silero_vad.onnx`，如果不存在会 fallback 到 `/home/urie/temp/silero-vad/src/silero_vad/data/silero_vad.onnx`（本机开发专用）。

### 6.4 火山引擎 TTS pacing
- ESP32 `PushPacketToDecodeQueue(false)` 在解码队列满时会**丢包**。所以不能 firehose。
- 但固定每包 sleep 60ms 也不行——火山常用 20ms 包，固定 60ms 会欠采样、听起来一顿一顿。
- 实现采用 `xiaozhi-server-go` 的策略：
  - 解析每个 Opus 包的 TOC 字节得到真实时长（`opus_duration.rs`）
  - 默认预缓冲 180ms（`VOLCENGINE_TTS_PREBUFFER_MS`），快速发
  - 之后按 `play_position - prebuffer` 时间轴调度（不是固定 sleep）
  - 最后 `sleep(min(prebuffer, play_position))` 再发 `tts.stop`
- 出问题先调 `VOLCENGINE_TTS_PREBUFFER_MS`（180 / 240 / 300），别改别的。

### 6.5 火山引擎 TTS voice ↔ resource_id 绑定
- `derive_resource_id` 从 voice_type 自动推断 resource_id：
  - `S_*` / `s_*` → `seed-icl-2.0`（声音克隆）
  - `*_uranus_bigtts` / `saturn_*` → `seed-tts-2.0`
  - `*_moon_bigtts` / `*_mars_bigtts` / `ICL_*` → `seed-tts-1.0`
- 错配会返回 `55000000: resource ID is mismatched`。如果用了新声音报这个错，先查 voice 列表确认前缀。
- TTS 输出是 `ogg_opus`，必须 `OggOpusPacketizer` demux（已在 `volcengine.rs` 里完成）。

### 6.6 流式 vs 非流式 ASR/LLM
- ASR：流式喂音频（降低首字延迟），但**只返回最终文本**。中间不发。
- LLM：流式输出文本给 TTS。`text_delta` 进 TTS，`thinking_*` 丢弃。
- TTS：流式把 LLM 文本喂进去（按 5ms 一个字），流式吐出 Opus 帧。

### 6.7 listening 状态外的音频
`handle_binary` 在 `!listening` 时**直接丢弃**二进制帧（连 trace 日志都不打，免得刷屏）。这是 ESP32 端偶尔会发的边角情况。

---

## 7. 环境变量速查

按用途分组。优先级：本项目 `.envrc` → 用户 shell → `Config::from_env()` 默认值。

### 服务器
| 变量 | 默认 | 说明 |
|------|------|------|
| `XIAOZHI_BIND` | `0.0.0.0:8080` | HTTP/WS 监听地址 |
| `XIAOZHI_PUBLIC_WS_URL` | `ws://127.0.0.1:<port>/ws` | `/ota` 返回给设备的 WS 地址；ESP32 必须能访问（局域网 IP，不是 127.0.0.1） |
| `XIAOZHI_TOKEN` | `dev-token` | OTA/WS 鉴权 token；空字符串 = 不校验 |
| `XIAOZHI_LISTEN_MAX_TIMEOUT_MS` | `120000` | 单次 listen 最长时长 |
| `XIAOZHI_IDLE_CLOSE_SECONDS` | `90` | 设备空闲多久后，模拟用户发"告别 prompt"让 LLM 自然收尾；0 / 负数 / 非法值回落到默认 |
| `XIAOZHI_EXIT_COMMANDS` | `退出,关闭` | ASR 最终文本精确命中这些关键词（去除首尾空白 + 常见标点）后，标记 `close_after_chat=true`；空串或未设置走默认。分隔符只有 `,` 和 `、` |
| `XIAOZHI_END_PROMPT` | `请你以"时间过得真快"开头，用富有感情、依依不舍的话来结束这场对话吧！` | idle watchdog 触发时喂给 LLM 的合成用户输入 |
| `XIAOZHI_END_PROMPT_ENABLED` | `true` | 设 `0`/`false`/`off`/`no`/`disabled` 可整体关闭 watchdog |

### ASR
| 变量 | 默认 | 说明 |
|------|------|------|
| `XIAOZHI_ASR_PROVIDER` | `mock` | `mock` / `volcengine` / `volc` / `doubao` |
| `VOLCENGINE_ASR_API_KEY` 或 `VOLCENGINE_API_KEY` | 必填（volc） | 火山引擎 ASR API key |
| `VOLCENGINE_ASR_RESOURCE_ID` | `volc.bigasr.sauc.duration` | |
| `VOLCENGINE_ASR_ENDPOINT` | `wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async` | |
| `VOLCENGINE_ASR_LANGUAGE` | `zh-CN` | |
| `VOLCENGINE_ASR_CHUNK_MS` | `180` | 推 PCM 的 chunk 大小 |

### LLM
| 变量 | 默认 | 说明 |
|------|------|------|
| `XIAOZHI_LLM_PROVIDER` | `mock` | `mock` / `openai` / `pi` / `pi-rpc` |
| **pi 模式专属** | | |
| `XIAOZHI_PI_RPC_COMMAND` | `pi` | 可执行文件名 |
| `XIAOZHI_PI_RPC_FLAGS` | 安全默认值（见 §3.3） | **空格分隔**，完全透传给 `pi` |
| `XIAOZHI_PI_RPC_SYSTEM_PROMPT_FILE` | 必填（如果要用） | 文件路径；内容会拼成 `--system-prompt <text>` |
| `XIAOZHI_PI_RPC_CWD` | 继承 | 子进程 cwd |
| `XIAOZHI_PI_RPC_IDLE_TIMEOUT_MS` | `300000` | stdout 静默多久杀子进程 |
| **OpenAI 兼容模式专属** | | |
| `XIAOZHI_LLM_API_KEY` / `OPENAI_API_KEY` | 必填 | |
| `XIAOZHI_LLM_MODEL` / `OPENAI_MODEL` | `gpt-4o-mini` | |
| `XIAOZHI_LLM_BASE_URL` / `OPENAI_BASE_URL` / `OPENAI_API_BASE` | `https://api.openai.com/v1` | |
| `XIAOZHI_LLM_SYSTEM_PROMPT` / `OPENAI_SYSTEM_PROMPT` | — | |
| `XIAOZHI_LLM_TEMPERATURE` / `OPENAI_TEMPERATURE` | — | |
| `XIAOZHI_LLM_MAX_TOKENS` / `OPENAI_MAX_TOKENS` | — | |
| `XIAOZHI_LLM_DISABLE_THINKING` | `true` | 默认禁用 thinking |
| `XIAOZHI_LLM_THINKING_STYLE` | `auto` | `auto` / `deepseek` / `enable_thinking` / `both` / `none`；auto 根据 base_url 推断 |

### TTS
| 变量 | 默认 | 说明 |
|------|------|------|
| `XIAOZHI_TTS_PROVIDER` | `mock` | `mock` / `volcengine` / `volc` |
| `VOLCENGINE_TTS_API_KEY` 或 `VOLCENGINE_API_KEY` | 必填（volc） | |
| `VOLCENGINE_TTS_VOICE_TYPE` | 必填（volc） | 声音 ID；用于推断 resource_id |
| `VOLCENGINE_TTS_ENDPOINT` | `wss://openspeech.bytedance.com/api/v3/tts/bidirection` | |
| `VOLCENGINE_TTS_ENCODING` | `ogg_opus` | `ogg_opus` 或 `raw_opus` |
| `VOLCENGINE_TTS_PREBUFFER_MS` | `180` | 火山 TTS pacing 预缓冲 |

### VAD
| 变量 | 默认 | 说明 |
|------|------|------|
| `XIAOZHI_VAD_PROVIDER` | `silero` | `silero` / `none` / `off` / `disabled` |
| `SILERO_VAD_MODEL_PATH` / `XIAOZHI_VAD_MODEL_PATH` | `models/silero_vad.onnx` | |
| `XIAOZHI_VAD_THRESHOLD` | `0.5` | 触发阈值 |
| `XIAOZHI_VAD_MIN_SILENCE_MS` | `600` | 静音多少 ms 才算句末 |
| `XIAOZHI_VAD_SPEECH_PAD_MS` | `64` | |
| `XIAOZHI_VAD_MIN_SPEECH_MS` | `160` | 短于这个不算 |
| `XIAOZHI_VAD_MAX_SPEECH_SECONDS` | `15.0` | 单段最长；超过强制结束 |

### ONNX Runtime
| 变量 | 默认 | 说明 |
|------|------|------|
| `ORT_DYLIB_PATH` | `${pkgs.onnxruntime}/lib/libonnxruntime.so`（Nix 提供） | 启动期必须存在 |
| `ORT_SKIP_DOWNLOAD` | `1` | Nix 环境固定 |

---

## 8. 编译 / 运行

### 首次进入
```bash
cd /home/urie/workspace/rust/xiaozhi-server-rs
direnv reload         # 或: nix-shell
```

### 检查 & 测试
```bash
cargo fmt
cargo check
cargo test
```

### 启动（默认走 `.envrc`）
```bash
nix-shell --run 'cargo run'
# 或单独 shell 里:
direnv exec . cargo run
```

### 跟 ESP32 连通性测试
- 确认 `XIAOZHI_PUBLIC_WS_URL` 用的是**电脑局域网 IP**（如 `ws://192.168.2.1:8080/ws`），不能用 `127.0.0.1`。
- ESP32 端 NVS 写入 token 后会带到 `Authorization` header；没配置时本服务**目前放行**。
- 跑起来后看 `RUST_LOG=xiaozhi_server_rs=debug,tower_http=info`，关键日志：
  - `using volcengine tts / asr / pi rpc llm / Silero VAD` — provider 选择
  - `websocket session connected` — 客户端连上
  - `using pi rpc llm command=... rendered_args=...` — pi 命令行（看 flag 透传是否对）
  - `client hello received version=V2/V3/V1` — 二进制协议版本
  - `listen started vad_enabled=true` — VAD 启用
  - `VAD speech started; opening ASR` — VAD 触发
  - `VAD speech ended` / `listen finished reason=...` — listen 结束
  - `starting conversation pipeline` → `first llm text chunk` → `first tts sentence_start` → `first tts audio frame ready` — 全链路首字时延
  - `pipeline finished / aborted` — 收尾

### 想看每个 Opus 包的 pacing
临时打开：
```bash
RUST_LOG=xiaozhi_server_rs::services::volcengine=trace cargo run
```
但**不要常驻**，trace 级别刷屏严重。

---

## 9. 测试现状

- `cargo test` 覆盖：
  - `protocol.rs`：JSON 解析、binary v1/v2/v3 roundtrip
  - `ogg_opus.rs`：demux 单元测试
  - `opus_duration.rs`：TOC 字节解析
  - `volcengine_asr.rs`：extract_text、消息头
  - `silero_vad.rs`：状态机（含连续 5 帧去抖、瞬态噪声过滤）
  - `openai.rs`：SSE delta 解析、thinking config 注入
  - `text_filter.rs`：markdown/emoji/paren 跨 chunk 过滤

- **没有**端到端测试。集成靠 `cargo check` + 手动 ESP32 联调。
- 加新 provider 时请至少补：配置加载 + 一个 happy-path 单元测试。

---

## 10. 改动指南（常见任务）

### 加一个新的 LLM provider
1. `src/services/<name>.rs` 实现 `LlmSessionFactory + LlmSession`（注意 §3.2 的同步约定）。
2. 在 `services/mod.rs::ServiceBundle::from_env()` 加分支，env 变量名沿用 `XIAOZHI_LLM_PROVIDER=<name>`。
3. 写测试，跑 `cargo check && cargo test`。

### 加一个新的 TTS provider
1. `src/services/<name>.rs` 实现 `TtsService`，如果输出不是裸 Opus，要自带 demuxer（参考 `ogg_opus.rs`）。
2. 注意 pacing：ESP32 客户端解码队列有上限，不能 firehose；优先用 `opus_duration::packet_duration` 做时间轴调度。

### 加一个新的 ASR provider
1. `src/services/<name>.rs` 实现 `AsrService + AsrStream`，**接收 PCM i16**，不要让上层再解码一次。
2. 如果需要分块上传，先 buffer 满 `chunk_bytes` 再发（参考 `volcengine_asr.rs::flush_ready_chunks`）。

### 改 VAD 阈值
- 默认值在 `silero_vad.rs::from_env()`。生产调优一般调：
  - `XIAOZHI_VAD_THRESHOLD`（环境噪声大 → 调到 0.55~0.65）
  - `XIAOZHI_VAD_MIN_SILENCE_MS`（希望快速响应 → 调到 300~500）
- 不要改 `REQUIRED_CONSECUTIVE_FRAMES`（去抖常数），除非你能证明新场景需要。

### 改协议
- `protocol.rs` 的常量（`SERVER_SAMPLE_RATE` 等）和二进制布局改完一定要同步更新 `/home/urie/workspace/cpp/xiaozhi-esp32/PROTOCOL.md`。
- 加新 JSON 消息类型先在 `IncomingJson.extra` 里读，不要加新字段除非你打算拒绝老客户端。

---

## 11. 已知限制 / 后续 TODO

- **没有持久化**：每个 WebSocket 连接都新建 session，设备重连 = 新对话历史（也包括 pi 子进程）。
- **没有 MCP**：v1 直接忽略 `type:"mcp"`。
- **没有认证强制**：`XIAOZHI_TOKEN` 缺失时放行；上线前请改成强制。
- **没有端到端测试**：目前只靠手动联调。
- **没有 rate limit / 单设备并发控制**。
- **`/`ota` 是固定响应**：不分设备、不查 NVS、不真升级。
- **pi-rpc 历史**：当前每个 prompt 走独立 request，没在 `chat_stream` 之间复用 conversation 上下文（这是 §3.2 简化决定的）。
- **OpenAI LLM 没有 abort 协议支持**：abort 路径只是 `let _ = self.llm.abort()`，OpenAI provider 实际没有 abort 流；要真打断得换 provider 或改成 SSE-aware。

### 11.1 退出 / 退场逻辑（按 Python `xiaozhi-esp32-server` 的简化版）

session.rs 实现了两条触发路径，共同复用 `SessionState.close_after_chat` 标志 + `SessionContext.shutdown_tx`：

1. **硬关键字匹配**（`XIAOZHI_EXIT_COMMANDS`，默认 `["退出","关闭"]`）
   - `run_pipeline_with_text` 入口对 ASR 返回文本做精确等值比对（容许首尾空白 + 中英文 `。！？…~`）
   - 命中：设 `close_after_chat=true`，**仍然把原文本喂给 LLM** 让它生成告别语
   - 不跳过 LLM/TTS，保证用户听到自然告别

2. **空闲超时 watchdog**（`XIAOZHI_IDLE_CLOSE_SECONDS`，默认 90s）
   - `handle_websocket` 启动一个 5s 轮询任务
   - 触发条件：`!listening && pipeline.is_none() && last_voice_ts.elapsed() >= 阈值 && !close_after_chat`
   - 触发后**原子抢占**：设 `close_after_chat=true` + 把 `last_voice_ts` 重置，然后用 `XIAOZHI_END_PROMPT` 作为合成用户输入喂 `run_pipeline_with_text`

**TTS 收尾 → 关闭 WS**（`maybe_schedule_close_after_chat`）
- 在 `tts.stop` 之后检查：
  - `aborted=true` → **不**关（被打断说明用户又不想退了，跟 Python 行为一致）
  - `close_after_chat=true` → `sleep(POST_TTS_FLUSH_DELAY=240ms)`（让最后一帧到 ESP32 并播放完）→ `ctx.shutdown_tx.send(())`
- `handle_websocket` 主循环 `tokio::select!` 收到 shutdown 后 break → 自然清理

**已知 race**（未修复）：
- watchdog 触发与 listen.start 几乎同时发生时，listen.start 会走 `abort_active` 把 in-flight pipeline abort；aborted=true 后会跳过关 WS，最终用户继续对话但不关 WS，直到下一轮 idle 再触发。这是简化版的取舍。

---

## 12. 参考资料

| 主题 | 路径 |
|------|------|
| 协议规范 | `/home/urie/workspace/cpp/xiaozhi-esp32/PROTOCOL.md` |
| ESP32 客户端 main | `/home/urie/workspace/cpp/xiaozhi-esp32/main/application.cc` |
| WebSocket 协议实现 | `/home/urie/workspace/cpp/xiaozhi-esp32/main/protocols/websocket_protocol.cc` |
| Go 参考服务 | `/home/urie/workspace/go/xiaozhi-server-go/`（特别是 `core/connection_sendmsg.go` 看 pacing，`core/providers/asr/doubao/doubao.go` 看 ASR 二进制协议） |
| 火山引擎 demo | `/home/urie/workspace/rust/volcengine_bidirection_demo/` |
| Silero VAD 参考 | `/home/urie/temp/silero-vad/examples/rust-example/` |
| Pi RPC 协议 | `/nix/store/qa1nry4gm37zs9s0qb1yrl2ka7k9jzfp-pi-0.79.9/lib/node_modules/@earendil-works/pi-coding-agent/docs/rpc.md` |

---

## 13. 一句话契约

> 每个 WebSocket 连接 = 一个独立 session；LLM 长生命周期、ASR/VAD/Pipeline 短生命周期；
> pipeline 永远用 `Notify` 退出而非 `abort`；Opus 共享解码、时间轴调度、按需 demux；
> 启动期就完成所有重资源加载（VAD 模型 / ONNX Runtime / rustls crypto provider）。