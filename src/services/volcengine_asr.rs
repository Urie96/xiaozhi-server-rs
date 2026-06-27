use std::{io::Write, time::Duration};

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use futures_util::{SinkExt, StreamExt, stream::SplitSink};
use serde_json::{Value, json};
use tokio::{net::TcpStream, task::JoinHandle, time::timeout};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async,
    tungstenite::{Message as WsMessage, client::IntoClientRequest},
};
use uuid::Uuid;

use crate::{
    audio::opus_decode::pcm_i16_to_le_bytes,
    services::{AsrService, AsrStream},
};

const DEFAULT_ENDPOINT: &str = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async";
const DEFAULT_RESOURCE_ID: &str = "volc.bigasr.sauc.duration";
const CLIENT_FULL_REQUEST: u8 = 0x1;
const CLIENT_AUDIO_REQUEST: u8 = 0x2;
const SERVER_FULL_RESPONSE: u8 = 0x9;
const SERVER_ERROR_RESPONSE: u8 = 0xf;
const FLAG_NO_SEQUENCE: u8 = 0x0;
const FLAG_LAST_PACKAGE: u8 = 0x2;
const SERIALIZATION_NONE: u8 = 0x0;
const SERIALIZATION_JSON: u8 = 0x1;
const COMPRESSION_GZIP: u8 = 0x1;
const ASR_SAMPLE_RATE: u32 = 16_000;

#[derive(Clone, Debug)]
pub struct VolcengineAsrConfig {
    pub api_key: Option<String>,
    pub app_id: Option<String>,
    pub access_key: Option<String>,
    pub resource_id: String,
    pub endpoint: String,
    pub language: String,
    pub chunk_ms: u32,
    pub enable_itn: bool,
    pub enable_punc: bool,
    pub enable_ddc: bool,
}

impl VolcengineAsrConfig {
    pub fn from_env() -> Result<Option<Self>> {
        let provider = std::env::var("XIAOZHI_ASR_PROVIDER")
            .unwrap_or_else(|_| "mock".to_string())
            .to_ascii_lowercase();
        if provider != "volcengine" && provider != "volc" && provider != "doubao" {
            return Ok(None);
        }

        let api_key = env_any(&["VOLCENGINE_ASR_API_KEY", "VOLCENGINE_API_KEY"]);
        let app_id = env_any(&["VOLCENGINE_ASR_APP_ID", "VOLCENGINE_APP_ID"]);
        let access_key = env_any(&[
            "VOLCENGINE_ASR_ACCESS_KEY",
            "VOLCENGINE_ASR_ACCESS_TOKEN",
            "VOLCENGINE_ACCESS_TOKEN",
        ]);

        if api_key.is_none() && (app_id.is_none() || access_key.is_none()) {
            bail!(
                "missing Volcengine ASR credentials; set VOLCENGINE_ASR_API_KEY or VOLCENGINE_ASR_APP_ID + VOLCENGINE_ASR_ACCESS_KEY"
            );
        }

        Ok(Some(Self {
            api_key,
            app_id,
            access_key,
            resource_id: env_or("VOLCENGINE_ASR_RESOURCE_ID", DEFAULT_RESOURCE_ID),
            endpoint: env_or("VOLCENGINE_ASR_ENDPOINT", DEFAULT_ENDPOINT),
            language: env_or("VOLCENGINE_ASR_LANGUAGE", "zh-CN"),
            chunk_ms: env_u32("VOLCENGINE_ASR_CHUNK_MS", 180).clamp(60, 400),
            enable_itn: env_bool("VOLCENGINE_ASR_ENABLE_ITN", true),
            enable_punc: env_bool("VOLCENGINE_ASR_ENABLE_PUNC", true),
            enable_ddc: env_bool("VOLCENGINE_ASR_ENABLE_DDC", false),
        }))
    }
}

#[derive(Clone, Debug)]
pub struct VolcengineAsr {
    config: VolcengineAsrConfig,
}

impl VolcengineAsr {
    pub fn new(config: VolcengineAsrConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl AsrService for VolcengineAsr {
    async fn start_stream(&self) -> Result<Box<dyn AsrStream>> {
        let stream = VolcengineAsrStream::connect(self.config.clone()).await?;
        Ok(Box::new(stream))
    }
}

type AsrWrite = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, WsMessage>;

pub struct VolcengineAsrStream {
    config: VolcengineAsrConfig,
    write: Option<AsrWrite>,
    reader: Option<JoinHandle<Result<String>>>,
    pcm_buffer: BytesMut,
    chunk_bytes: usize,
    finished: bool,
}

impl VolcengineAsrStream {
    async fn connect(config: VolcengineAsrConfig) -> Result<Self> {
        let mut request = config
            .endpoint
            .as_str()
            .into_client_request()
            .context("build volcengine asr websocket request")?;

        if let Some(api_key) = &config.api_key {
            request
                .headers_mut()
                .insert("X-Api-Key", api_key.parse().context("invalid ASR api key")?);
        } else {
            request.headers_mut().insert(
                "X-Api-App-Key",
                config
                    .app_id
                    .as_deref()
                    .unwrap_or_default()
                    .parse()
                    .context("invalid ASR app id")?,
            );
            request.headers_mut().insert(
                "X-Api-Access-Key",
                config
                    .access_key
                    .as_deref()
                    .unwrap_or_default()
                    .parse()
                    .context("invalid ASR access key")?,
            );
        }
        request.headers_mut().insert(
            "X-Api-Resource-Id",
            config
                .resource_id
                .parse()
                .context("invalid ASR resource id")?,
        );
        let request_id = Uuid::new_v4().to_string();
        request
            .headers_mut()
            .insert("X-Api-Connect-Id", request_id.parse()?);
        request
            .headers_mut()
            .insert("X-Api-Request-Id", request_id.parse()?);
        request
            .headers_mut()
            .insert("X-Api-Sequence", "-1".parse()?);

        let auth_mode = if config.api_key.is_some() {
            "api_key"
        } else {
            "app_id_access_key"
        };
        let (ws, response) = connect_async(request).await.with_context(|| {
            format!(
                "connect volcengine asr websocket endpoint={} resource_id={} auth_mode={}",
                config.endpoint, config.resource_id, auth_mode
            )
        })?;
        tracing::info!(
            status = %response.status(),
            log_id = ?response.headers().get("x-tt-logid"),
            endpoint = %config.endpoint,
            resource_id = %config.resource_id,
            "volcengine asr websocket connected"
        );

        let (mut write, mut read) = ws.split();
        let request_payload = serde_json::to_vec(&json!({
            "user": {
                "uid": Uuid::new_v4().to_string(),
            },
            "audio": {
                "format": "pcm",
                "rate": ASR_SAMPLE_RATE,
                "bits": 16,
                "channel": 1,
                "language": &config.language,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": config.enable_itn,
                "enable_punc": config.enable_punc,
                "enable_ddc": config.enable_ddc,
                "result_type": "single",
                "show_utterances": false,
            }
        }))?;
        let init_message = AsrMessage::client_full(gzip(&request_payload)?).encode();
        write
            .send(WsMessage::Binary(init_message))
            .await
            .context("send volcengine ASR init request")?;

        let init = wait_initial_response(&mut read).await?;
        validate_initial_response(&init)?;

        let reader = tokio::spawn(async move { read_asr_results(read).await });
        let chunk_bytes = (ASR_SAMPLE_RATE as usize * config.chunk_ms as usize / 1000) * 2;

        Ok(Self {
            config,
            write: Some(write),
            reader: Some(reader),
            pcm_buffer: BytesMut::new(),
            chunk_bytes,
            finished: false,
        })
    }

    async fn send_pcm_chunk(&mut self, chunk: Bytes, is_last: bool) -> Result<()> {
        let Some(write) = self.write.as_mut() else {
            bail!("volcengine ASR stream is closed");
        };
        let message = AsrMessage::audio(gzip(&chunk)?, is_last).encode();
        write
            .send(WsMessage::Binary(message))
            .await
            .context("send volcengine ASR audio")
    }

    async fn flush_ready_chunks(&mut self) -> Result<()> {
        while self.pcm_buffer.len() >= self.chunk_bytes {
            let chunk = self.pcm_buffer.split_to(self.chunk_bytes).freeze();
            self.send_pcm_chunk(chunk, false).await?;
        }
        Ok(())
    }
}

#[async_trait]
impl AsrStream for VolcengineAsrStream {
    async fn push_pcm(&mut self, samples: &[i16]) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        let pcm = pcm_i16_to_le_bytes(samples);
        self.pcm_buffer.extend_from_slice(&pcm);
        self.flush_ready_chunks().await
    }

    async fn finish(&mut self) -> Result<String> {
        if self.finished {
            return Ok(String::new());
        }
        self.finished = true;

        let final_chunk = if self.pcm_buffer.is_empty() {
            Bytes::new()
        } else {
            self.pcm_buffer.split().freeze()
        };
        self.send_pcm_chunk(final_chunk, true).await?;
        self.write.take();

        let Some(reader) = self.reader.take() else {
            return Ok(String::new());
        };
        let text = timeout(Duration::from_secs(15), reader)
            .await
            .context("timeout waiting for final volcengine ASR result")?
            .context("volcengine ASR reader task panicked")??;
        tracing::info!(text, "volcengine asr finalized");
        Ok(text)
    }

    async fn abort(&mut self) {
        self.finished = true;
        self.write.take();
        if let Some(reader) = self.reader.take() {
            reader.abort();
        }
        tracing::debug!(
            endpoint = %self.config.endpoint,
            "volcengine asr stream aborted"
        );
    }
}

impl Drop for VolcengineAsrStream {
    fn drop(&mut self) {
        if let Some(reader) = self.reader.take() {
            reader.abort();
        }
    }
}

async fn wait_initial_response<R>(read: &mut R) -> Result<AsrResponse>
where
    R: futures_util::Stream<
            Item = std::result::Result<WsMessage, tokio_tungstenite::tungstenite::Error>,
        > + Unpin,
{
    loop {
        let Some(message) = timeout(Duration::from_secs(15), read.next())
            .await
            .context("timeout waiting for volcengine ASR init response")?
        else {
            bail!("volcengine ASR websocket closed before init response");
        };
        let message = message.context("read volcengine ASR init response")?;
        let Some(response) = decode_ws_message(message)? else {
            continue;
        };
        return Ok(response);
    }
}

fn validate_initial_response(response: &AsrResponse) -> Result<()> {
    if response.message_type == SERVER_ERROR_RESPONSE {
        bail!(
            "volcengine ASR init error {}: {}",
            response.error_code,
            response.payload
        );
    }

    if let Some(code) = response.payload.get("code").and_then(|v| v.as_i64()) {
        if code != 20000000 && code != 1000 {
            bail!("volcengine ASR init failed: {}", response.payload);
        }
    }
    Ok(())
}

async fn read_asr_results(
    mut read: futures_util::stream::SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>,
) -> Result<String> {
    let mut latest_text = String::new();

    loop {
        let Some(message) = timeout(Duration::from_secs(60), read.next())
            .await
            .context("timeout waiting for volcengine ASR response")?
        else {
            return Ok(latest_text);
        };
        let message = message.context("read volcengine ASR websocket")?;
        let Some(response) = decode_ws_message(message)? else {
            continue;
        };

        if response.message_type == SERVER_ERROR_RESPONSE {
            bail!(
                "volcengine ASR error {}: {}",
                response.error_code,
                response.payload
            );
        }

        tracing::trace!(
            is_last = response.is_last_package,
            payload = %response.payload,
            "volcengine asr response"
        );

        if let Some(text) = extract_text(&response.payload) {
            if !text.is_empty() {
                latest_text = text;
            }
        }

        if response.is_last_package {
            return Ok(latest_text);
        }
    }
}

fn decode_ws_message(message: WsMessage) -> Result<Option<AsrResponse>> {
    match message {
        WsMessage::Binary(data) => Ok(Some(AsrResponse::decode(data)?)),
        WsMessage::Text(text) => Ok(Some(AsrResponse::decode(Bytes::copy_from_slice(
            text.as_bytes(),
        ))?)),
        WsMessage::Ping(_) | WsMessage::Pong(_) | WsMessage::Frame(_) => Ok(None),
        WsMessage::Close(frame) => bail!("volcengine ASR websocket closed: {frame:?}"),
    }
}

#[derive(Debug)]
struct AsrMessage {
    message_type: u8,
    flags: u8,
    serialization: u8,
    compression: u8,
    payload: Bytes,
}

impl AsrMessage {
    fn client_full(payload: Bytes) -> Self {
        Self {
            message_type: CLIENT_FULL_REQUEST,
            flags: FLAG_NO_SEQUENCE,
            serialization: SERIALIZATION_JSON,
            compression: COMPRESSION_GZIP,
            payload,
        }
    }

    fn audio(payload: Bytes, is_last: bool) -> Self {
        Self {
            message_type: CLIENT_AUDIO_REQUEST,
            flags: if is_last {
                FLAG_LAST_PACKAGE
            } else {
                FLAG_NO_SEQUENCE
            },
            serialization: SERIALIZATION_NONE,
            compression: COMPRESSION_GZIP,
            payload,
        }
    }

    fn encode(self) -> Bytes {
        let mut out = BytesMut::with_capacity(8 + self.payload.len());
        out.put_u8(0x11); // version 1, header size 1 word (4 bytes)
        out.put_u8((self.message_type << 4) | self.flags);
        out.put_u8((self.serialization << 4) | self.compression);
        out.put_u8(0);
        out.put_u32(self.payload.len() as u32);
        out.extend_from_slice(&self.payload);
        out.freeze()
    }
}

#[derive(Debug)]
struct AsrResponse {
    message_type: u8,
    is_last_package: bool,
    payload: Value,
    error_code: u32,
}

impl AsrResponse {
    fn decode(mut data: Bytes) -> Result<Self> {
        if data.len() < 4 {
            bail!("volcengine ASR message too short: {}", data.len());
        }

        let version_and_header = data.get_u8();
        let header_size = ((version_and_header & 0x0f) as usize) * 4;
        let type_and_flags = data.get_u8();
        let message_type = type_and_flags >> 4;
        let flags = type_and_flags & 0x0f;
        let serialization_and_compression = data.get_u8();
        let serialization = serialization_and_compression >> 4;
        let compression = serialization_and_compression & 0x0f;
        data.get_u8();

        if header_size < 4 {
            bail!("invalid volcengine ASR header size: {header_size}");
        }
        if header_size > 4 {
            if data.len() < header_size - 4 {
                bail!("volcengine ASR header extension truncated");
            }
            data.advance(header_size - 4);
        }

        let is_last_package = flags & FLAG_LAST_PACKAGE != 0;
        let mut error_code = 0;

        if message_type == SERVER_ERROR_RESPONSE {
            if data.len() < 8 {
                bail!("volcengine ASR error response truncated");
            }
            error_code = data.get_u32();
            let payload = read_payload(&mut data, serialization, compression)?;
            return Ok(Self {
                message_type,
                is_last_package,
                payload,
                error_code,
            });
        }

        if message_type != SERVER_FULL_RESPONSE {
            return Ok(Self {
                message_type,
                is_last_package,
                payload: Value::Null,
                error_code,
            });
        }

        if flags & 0x01 != 0 {
            if data.len() < 4 {
                bail!("volcengine ASR response missing sequence");
            }
            let _sequence = data.get_i32();
        }

        if flags & 0x04 != 0 {
            if data.len() < 4 {
                bail!("volcengine ASR response missing event");
            }
            let _event = data.get_i32();
        }

        let payload = read_payload(&mut data, serialization, compression)?;
        Ok(Self {
            message_type,
            is_last_package,
            payload,
            error_code,
        })
    }
}

fn read_payload(data: &mut Bytes, serialization: u8, compression: u8) -> Result<Value> {
    if data.len() < 4 {
        bail!("volcengine ASR response missing payload size");
    }
    let payload_size = data.get_u32() as usize;
    if data.len() < payload_size {
        bail!(
            "volcengine ASR payload truncated: declared {}, available {}",
            payload_size,
            data.len()
        );
    }
    let mut payload = data.copy_to_bytes(payload_size).to_vec();
    if compression == COMPRESSION_GZIP {
        let mut decoder = GzDecoder::new(&payload[..]);
        let mut decoded = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut decoded)
            .context("decompress volcengine ASR response")?;
        payload = decoded;
    }

    if serialization == SERIALIZATION_JSON {
        serde_json::from_slice(&payload).context("parse volcengine ASR response json")
    } else if payload.is_empty() {
        Ok(Value::Null)
    } else {
        Ok(Value::String(String::from_utf8_lossy(&payload).to_string()))
    }
}

fn extract_text(payload: &Value) -> Option<String> {
    let result = payload.get("result")?;
    if let Some(text) = result.get("text").and_then(|v| v.as_str()) {
        return Some(text.to_string());
    }

    // Some variants can return result as a list. Keep the last non-empty text.
    if let Some(items) = result.as_array() {
        return items
            .iter()
            .filter_map(|item| item.get("text").and_then(|v| v.as_str()))
            .filter(|text| !text.is_empty())
            .last()
            .map(ToOwned::to_owned);
    }

    None
}

fn gzip(data: &[u8]) -> Result<Bytes> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(data)
        .context("gzip volcengine ASR data")?;
    Ok(Bytes::from(encoder.finish().context("finish gzip")?))
}

fn env_any(names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|name| std::env::var(name).ok().filter(|s| !s.is_empty()))
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .and_then(|s| match s.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_object_result_text() {
        let payload = json!({"result":{"text":"你好"}});
        assert_eq!(extract_text(&payload).as_deref(), Some("你好"));
    }

    #[test]
    fn asr_message_header_uses_gzip_audio_without_json_serialization() {
        let msg = AsrMessage::audio(Bytes::from_static(b"abc"), true).encode();
        assert_eq!(msg[0], 0x11);
        assert_eq!(msg[1], 0x22);
        assert_eq!(msg[2], 0x01);
        assert_eq!(&msg[4..8], 3u32.to_be_bytes().as_slice());
    }
}
