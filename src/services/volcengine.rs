use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use async_stream::try_stream;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use futures_util::{Sink, SinkExt, Stream, StreamExt};
use serde_json::json;
use tokio::time::{sleep, timeout};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{Message as WsMessage, client::IntoClientRequest},
};
use uuid::Uuid;

use crate::{
    audio::ogg_opus::OggOpusPacketizer,
    protocol::{AudioFrame, SERVER_FRAME_DURATION_MS, SERVER_SAMPLE_RATE},
};

use super::{TextStream, TtsEvent, TtsService, TtsStream};

const DEFAULT_ENDPOINT: &str = "wss://openspeech.bytedance.com/api/v3/tts/bidirection";
const DEFAULT_RESOURCE_ID: &str = "seed-tts-2.0";

#[derive(Clone, Debug)]
pub struct VolcengineTtsConfig {
    pub api_key: String,
    pub resource_id: String,
    pub voice_type: String,
    pub endpoint: String,
    pub encoding: VolcengineAudioEncoding,
}

impl VolcengineTtsConfig {
    pub fn from_env() -> Result<Option<Self>> {
        let provider = std::env::var("XIAOZHI_TTS_PROVIDER")
            .unwrap_or_else(|_| "mock".to_string())
            .to_ascii_lowercase();
        if provider != "volcengine" && provider != "volc" {
            return Ok(None);
        }

        let voice_type = required_env("VOLCENGINE_TTS_VOICE_TYPE")?;
        let resource_id = env_or("VOLCENGINE_TTS_RESOURCE_ID", DEFAULT_RESOURCE_ID);

        Ok(Some(Self {
            api_key: required_env_any(&["VOLCENGINE_TTS_API_KEY", "VOLCENGINE_API_KEY"])?,
            resource_id,
            voice_type,
            endpoint: env_or("VOLCENGINE_TTS_ENDPOINT", DEFAULT_ENDPOINT),
            encoding: VolcengineAudioEncoding::from_env()?,
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolcengineAudioEncoding {
    /// Volcengine returns Ogg pages; the service demuxes them to raw Opus packets
    /// before forwarding to Xiaozhi clients.
    OggOpus,
    /// Advanced/debug mode. Payloads are forwarded as if they were raw Opus.
    RawOpus,
}

impl VolcengineAudioEncoding {
    fn from_env() -> Result<Self> {
        match env_or("VOLCENGINE_TTS_ENCODING", "ogg_opus").as_str() {
            "ogg_opus" | "ogg-opus" | "ogg" => Ok(Self::OggOpus),
            "opus" | "raw_opus" | "raw-opus" => Ok(Self::RawOpus),
            other => {
                bail!("unsupported VOLCENGINE_TTS_ENCODING={other}; expected ogg_opus or raw_opus")
            }
        }
    }

    fn request_format(self) -> &'static str {
        match self {
            Self::OggOpus => "ogg_opus",
            Self::RawOpus => "opus",
        }
    }
}

#[derive(Clone, Debug)]
pub struct VolcengineTts {
    config: VolcengineTtsConfig,
}

impl VolcengineTts {
    pub fn new(config: VolcengineTtsConfig) -> Self {
        Self { config }
    }
}

impl TtsService for VolcengineTts {
    fn synthesize_stream(&self, input: TextStream) -> TtsStream {
        let client = VolcengineBidirectionalClient::new(self.config.clone());
        Box::pin(try_stream! {
            let mut stream = client.synthesize(input).await?;
            while let Some(event) = stream.next().await {
                yield event?;
            }
        })
    }
}

struct VolcengineBidirectionalClient {
    config: VolcengineTtsConfig,
}

impl VolcengineBidirectionalClient {
    fn new(config: VolcengineTtsConfig) -> Self {
        Self { config }
    }

    async fn synthesize(self, input: TextStream) -> Result<TtsStream> {
        let mut request = self
            .config
            .endpoint
            .as_str()
            .into_client_request()
            .context("build volcengine tts websocket request")?;
        request.headers_mut().insert(
            "X-Api-Key",
            self.config
                .api_key
                .parse()
                .context("invalid api key header")?,
        );
        request.headers_mut().insert(
            "X-Api-Resource-Id",
            self.config
                .resource_id
                .parse()
                .context("invalid resource id header")?,
        );
        request
            .headers_mut()
            .insert("X-Api-Connect-Id", Uuid::new_v4().to_string().parse()?);
        request
            .headers_mut()
            .insert("X-Control-Require-Usage-Tokens-Return", "*".parse()?);

        let (ws, response) = connect_async(request)
            .await
            .context("connect volcengine bidirectional tts websocket")?;
        tracing::info!(
            status = %response.status(),
            log_id = ?response.headers().get("x-tt-logid"),
            "volcengine tts websocket connected"
        );

        let (mut write, mut read) = ws.split();

        send_volc_message(
            &mut write,
            VolcMessage::full_client(EventType::StartConnection, None, Bytes::from_static(b"{}")),
        )
        .await
        .context("send volcengine StartConnection")?;

        wait_for_event(
            &mut read,
            MsgType::FullServerResponse,
            EventType::ConnectionStarted,
        )
        .await
        .context("wait volcengine ConnectionStarted")?;

        let session_id = Uuid::new_v4().to_string();
        let request_template = self.request_template();
        let start_payload = serde_json::to_vec(&json!({
            "event": EventType::StartSession as i32,
            "req_params": request_template.req_params,
        }))?;

        send_volc_message(
            &mut write,
            VolcMessage::full_client(
                EventType::StartSession,
                Some(session_id.clone()),
                Bytes::from(start_payload),
            ),
        )
        .await
        .context("send volcengine StartSession")?;

        wait_for_event(
            &mut read,
            MsgType::FullServerResponse,
            EventType::SessionStarted,
        )
        .await
        .context("wait volcengine SessionStarted")?;

        let encoding = self.config.encoding;
        let writer_config = self.config;
        let writer_session_id = session_id.clone();
        let writer = AbortOnDrop::new(tokio::spawn(async move {
            write_text_stream(write, input, writer_config, writer_session_id).await
        }));

        let stream = try_stream! {
            let mut demuxer = OggOpusPacketizer::new();
            let mut timestamp = 0u32;

            loop {
                let Some(message) = timeout(Duration::from_secs(60), read.next())
                    .await
                    .context("timeout waiting for volcengine tts message")?
                else {
                    Err(anyhow!("volcengine tts websocket closed before SessionFinished"))?;
                    unreachable!();
                };
                let message = message.context("read volcengine tts websocket")?;
                let Some(volc_message) = decode_ws_message(message)? else {
                    continue;
                };

                match volc_message.msg_type {
                    MsgType::AudioOnlyServer => {
                        let packets = match encoding {
                            VolcengineAudioEncoding::OggOpus => demuxer.push(&volc_message.payload)?,
                            VolcengineAudioEncoding::RawOpus => vec![volc_message.payload],
                        };

                        for payload in packets {
                            yield TtsEvent::Audio(AudioFrame { timestamp, payload });
                            timestamp = timestamp.saturating_add(SERVER_FRAME_DURATION_MS);
                        }
                    }
                    MsgType::FullServerResponse => {
                        tracing::trace!(
                            event = ?volc_message.event,
                            payload = %String::from_utf8_lossy(&volc_message.payload),
                            "volcengine tts server response"
                        );
                        if volc_message.event == EventType::TtsSentenceStart {
                            yield TtsEvent::SentenceStart(String::new());
                        }
                        if volc_message.event == EventType::SessionFinished {
                            break;
                        }
                        if volc_message.event == EventType::SessionFailed {
                            Err(anyhow!(
                                "volcengine tts session failed: {}",
                                String::from_utf8_lossy(&volc_message.payload)
                            ))?;
                        }
                    }
                    MsgType::Error => {
                        Err(anyhow!(
                            "volcengine tts error {}: {}",
                            volc_message.error_code,
                            String::from_utf8_lossy(&volc_message.payload)
                        ))?;
                    }
                    other => tracing::trace!(msg_type = ?other, "ignored volcengine tts message"),
                }
            }

            match writer.join().await {
                Ok(Ok(())) => {}
                Ok(Err(err)) => Err(err)?,
                Err(err) => Err(anyhow!(err).context("volcengine tts writer task panicked"))?,
            }
        };

        Ok(Box::pin(stream))
    }

    fn request_template(&self) -> RequestTemplate {
        let req_params = json!({
            "speaker": &self.config.voice_type,
            "audio_params": {
                "format": self.config.encoding.request_format(),
                "sample_rate": SERVER_SAMPLE_RATE,
            },
        });

        RequestTemplate { req_params }
    }
}

struct RequestTemplate {
    req_params: serde_json::Value,
}

async fn write_text_stream<S>(
    mut write: S,
    mut input: TextStream,
    config: VolcengineTtsConfig,
    session_id: String,
) -> Result<()>
where
    S: Sink<WsMessage> + Unpin,
    <S as Sink<WsMessage>>::Error: std::error::Error + Send + Sync + 'static,
{
    while let Some(chunk) = input.next().await {
        let text = chunk?;
        if text.trim().is_empty() {
            continue;
        }

        for ch in text.chars() {
            let payload = serde_json::to_vec(&json!({
                "event": EventType::TaskRequest as i32,
                "req_params": {
                    "text": ch.to_string(),
                    "speaker": &config.voice_type,
                    "audio_params": {
                        "format": config.encoding.request_format(),
                        "sample_rate": SERVER_SAMPLE_RATE,
                    },
                },
            }))?;

            send_volc_message(
                &mut write,
                VolcMessage::full_client(
                    EventType::TaskRequest,
                    Some(session_id.clone()),
                    Bytes::from(payload),
                ),
            )
            .await
            .context("send volcengine TaskRequest")?;

            sleep(Duration::from_millis(5)).await;
        }
    }

    send_volc_message(
        &mut write,
        VolcMessage::full_client(
            EventType::FinishSession,
            Some(session_id),
            Bytes::from_static(b"{}"),
        ),
    )
    .await
    .context("send volcengine FinishSession")?;

    Ok(())
}

async fn send_volc_message<S>(write: &mut S, message: VolcMessage) -> Result<()>
where
    S: Sink<WsMessage> + Unpin,
    <S as Sink<WsMessage>>::Error: std::error::Error + Send + Sync + 'static,
{
    let bytes = message.encode()?;
    write
        .send(WsMessage::Binary(bytes))
        .await
        .context("send volcengine websocket message")
}

async fn wait_for_event<R>(read: &mut R, msg_type: MsgType, event: EventType) -> Result<VolcMessage>
where
    R: Stream<Item = std::result::Result<WsMessage, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    loop {
        let Some(message) = timeout(Duration::from_secs(15), read.next())
            .await
            .context("timeout waiting for volcengine event")?
        else {
            bail!("volcengine websocket closed while waiting for {event:?}");
        };
        let message = message.context("read volcengine websocket")?;
        let Some(message) = decode_ws_message(message)? else {
            continue;
        };

        if message.msg_type == MsgType::Error {
            bail!(
                "volcengine error {} while waiting for {event:?}: {}",
                message.error_code,
                String::from_utf8_lossy(&message.payload)
            );
        }

        if message.msg_type == msg_type && message.event == event {
            return Ok(message);
        }

        tracing::debug!(
            got_type = ?message.msg_type,
            got_event = ?message.event,
            want_type = ?msg_type,
            want_event = ?event,
            "skipping volcengine websocket event"
        );
    }
}

fn decode_ws_message(message: WsMessage) -> Result<Option<VolcMessage>> {
    match message {
        WsMessage::Binary(data) => Ok(Some(VolcMessage::decode(data)?)),
        WsMessage::Text(text) => Ok(Some(VolcMessage::decode(Bytes::copy_from_slice(
            text.as_bytes(),
        ))?)),
        WsMessage::Ping(_) | WsMessage::Pong(_) => Ok(None),
        WsMessage::Close(frame) => bail!("volcengine websocket closed: {frame:?}"),
        WsMessage::Frame(_) => Ok(None),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum MsgType {
    FullClientRequest = 0b0001,
    AudioOnlyServer = 0b1011,
    FullServerResponse = 0b1001,
    Error = 0b1111,
    Unknown = 0,
}

impl From<u8> for MsgType {
    fn from(value: u8) -> Self {
        match value {
            0b0001 => Self::FullClientRequest,
            0b1011 => Self::AudioOnlyServer,
            0b1001 => Self::FullServerResponse,
            0b1111 => Self::Error,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum MsgFlag {
    WithEvent = 0b0100,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
enum EventType {
    None = 0,
    StartConnection = 1,
    FinishConnection = 2,
    ConnectionStarted = 50,
    ConnectionFailed = 51,
    ConnectionFinished = 52,
    StartSession = 100,
    FinishSession = 102,
    SessionStarted = 150,
    SessionFinished = 152,
    SessionFailed = 153,
    TaskRequest = 200,
    TtsSentenceStart = 350,
}

struct AbortOnDrop<T> {
    handle: tokio::task::JoinHandle<T>,
}

impl<T> AbortOnDrop<T> {
    fn new(handle: tokio::task::JoinHandle<T>) -> Self {
        Self { handle }
    }

    async fn join(mut self) -> std::result::Result<T, tokio::task::JoinError> {
        (&mut self.handle).await
    }
}

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        if !self.handle.is_finished() {
            self.handle.abort();
        }
    }
}

impl From<i32> for EventType {
    fn from(value: i32) -> Self {
        match value {
            1 => Self::StartConnection,
            2 => Self::FinishConnection,
            50 => Self::ConnectionStarted,
            51 => Self::ConnectionFailed,
            52 => Self::ConnectionFinished,
            100 => Self::StartSession,
            102 => Self::FinishSession,
            150 => Self::SessionStarted,
            152 => Self::SessionFinished,
            153 => Self::SessionFailed,
            200 => Self::TaskRequest,
            350 => Self::TtsSentenceStart,
            _ => Self::None,
        }
    }
}

#[derive(Debug)]
struct VolcMessage {
    msg_type: MsgType,
    flag: u8,
    event: EventType,
    session_id: Option<String>,
    payload: Bytes,
    error_code: u32,
}

impl VolcMessage {
    fn full_client(event: EventType, session_id: Option<String>, payload: Bytes) -> Self {
        Self {
            msg_type: MsgType::FullClientRequest,
            flag: MsgFlag::WithEvent as u8,
            event,
            session_id,
            payload,
            error_code: 0,
        }
    }

    fn encode(self) -> Result<Bytes> {
        let mut out = BytesMut::new();
        out.put_u8(0x11); // version 1, header size 1 word (4 bytes)
        out.put_u8(((self.msg_type as u8) << 4) | self.flag);
        out.put_u8(0x10); // JSON serialization, no compression
        out.put_u8(0x00); // reserved / header padding

        if self.flag == MsgFlag::WithEvent as u8 {
            out.put_i32(self.event as i32);
            if event_has_session_id(self.event) {
                let session_id = self.session_id.unwrap_or_default();
                out.put_u32(session_id.len() as u32);
                out.extend_from_slice(session_id.as_bytes());
            }
        }

        out.put_u32(self.payload.len() as u32);
        out.extend_from_slice(&self.payload);
        Ok(out.freeze())
    }

    fn decode(mut data: Bytes) -> Result<Self> {
        if data.len() < 4 {
            bail!("volcengine message too short: {}", data.len());
        }

        let version_and_header = data.get_u8();
        let header_size = (version_and_header & 0x0f) as usize * 4;
        let type_and_flag = data.get_u8();
        let msg_type = MsgType::from(type_and_flag >> 4);
        let flag = type_and_flag & 0x0f;
        let _serialization_and_compression = data.get_u8();
        data.get_u8();

        if header_size > 4 {
            if data.len() < header_size - 4 {
                bail!("volcengine header extension truncated");
            }
            data.advance(header_size - 4);
        }

        let mut error_code = 0;
        if msg_type == MsgType::Error {
            if data.len() < 4 {
                bail!("volcengine error message missing error code");
            }
            error_code = data.get_u32();
        }

        if matches!(flag, 0b0001 | 0b0011) {
            if data.len() < 4 {
                bail!("volcengine sequenced message missing sequence");
            }
            let _sequence = data.get_i32();
        }

        let mut event = EventType::None;
        let mut session_id = None;
        if flag == MsgFlag::WithEvent as u8 {
            if data.len() < 4 {
                bail!("volcengine event message missing event id");
            }
            event = EventType::from(data.get_i32());

            if event_has_session_id(event) {
                session_id = Some(read_string(&mut data).context("read volcengine session id")?);
            }

            if matches!(
                event,
                EventType::ConnectionStarted
                    | EventType::ConnectionFailed
                    | EventType::ConnectionFinished
            ) {
                let _connect_id = read_string(&mut data).context("read volcengine connect id")?;
            }
        }

        if data.len() < 4 {
            bail!("volcengine message missing payload size");
        }
        let payload_size = data.get_u32() as usize;
        if data.len() < payload_size {
            bail!(
                "volcengine payload truncated: declared {}, available {}",
                payload_size,
                data.len()
            );
        }
        let payload = data.copy_to_bytes(payload_size);

        Ok(Self {
            msg_type,
            flag,
            event,
            session_id,
            payload,
            error_code,
        })
    }
}

fn read_string(data: &mut Bytes) -> Result<String> {
    if data.len() < 4 {
        bail!("string missing length");
    }
    let len = data.get_u32() as usize;
    if data.len() < len {
        bail!("string truncated: declared {len}, available {}", data.len());
    }
    let bytes = data.copy_to_bytes(len);
    String::from_utf8(bytes.to_vec()).context("invalid utf-8 string")
}

fn event_has_session_id(event: EventType) -> bool {
    !matches!(
        event,
        EventType::StartConnection
            | EventType::FinishConnection
            | EventType::ConnectionStarted
            | EventType::ConnectionFailed
            | EventType::ConnectionFinished
    )
}

fn required_env(name: &str) -> Result<String> {
    std::env::var(name)
        .ok()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("missing required env {name}"))
}

fn required_env_any(names: &[&str]) -> Result<String> {
    names
        .iter()
        .find_map(|name| std::env::var(name).ok().filter(|s| !s.is_empty()))
        .ok_or_else(|| anyhow!("missing required env; expected one of {}", names.join(", ")))
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default.to_string())
}
