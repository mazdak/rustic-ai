use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_write_tokens: u64,
    pub cache_read_tokens: u64,
    pub input_audio_tokens: u64,
    pub output_audio_tokens: u64,
    pub details: HashMap<String, u64>,
}

impl RequestUsage {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunUsage {
    pub requests: u64,
    pub tool_calls: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_write_tokens: u64,
    pub cache_read_tokens: u64,
    pub input_audio_tokens: u64,
    pub output_audio_tokens: u64,
    pub details: HashMap<String, u64>,
}

impl RunUsage {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    pub fn incr_request(&mut self, request: &RequestUsage) {
        self.requests += 1;
        self.input_tokens += request.input_tokens;
        self.output_tokens += request.output_tokens;
        self.cache_write_tokens += request.cache_write_tokens;
        self.cache_read_tokens += request.cache_read_tokens;
        self.input_audio_tokens += request.input_audio_tokens;
        self.output_audio_tokens += request.output_audio_tokens;
        for (k, v) in &request.details {
            *self.details.entry(k.clone()).or_insert(0) += v;
        }
    }

    pub fn incr_tool_call(&mut self) {
        self.tool_calls += 1;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageLimits {
    pub request_limit: Option<u64>,
    pub tool_calls_limit: Option<u64>,
    pub input_tokens_limit: Option<u64>,
    pub output_tokens_limit: Option<u64>,
    pub total_tokens_limit: Option<u64>,
}

impl Default for UsageLimits {
    fn default() -> Self {
        Self {
            request_limit: Some(50),
            tool_calls_limit: None,
            input_tokens_limit: None,
            output_tokens_limit: None,
            total_tokens_limit: None,
        }
    }
}

impl UsageLimits {
    pub fn check_request(&self, current_requests: u64) -> Result<(), UsageError> {
        if let Some(limit) = self.request_limit
            && current_requests >= limit
        {
            return Err(UsageError::RequestLimitExceeded { limit });
        }
        Ok(())
    }

    pub fn check_tool_call(&self, current_calls: u64) -> Result<(), UsageError> {
        if let Some(limit) = self.tool_calls_limit
            && current_calls >= limit
        {
            return Err(UsageError::ToolCallsLimitExceeded { limit });
        }
        Ok(())
    }

    pub fn check_after_response(&self, usage: &RunUsage) -> Result<(), UsageError> {
        if let Some(limit) = self.input_tokens_limit
            && usage.input_tokens > limit
        {
            return Err(UsageError::InputTokensLimitExceeded { limit });
        }
        if let Some(limit) = self.output_tokens_limit
            && usage.output_tokens > limit
        {
            return Err(UsageError::OutputTokensLimitExceeded { limit });
        }
        if let Some(limit) = self.total_tokens_limit
            && usage.total_tokens() > limit
        {
            return Err(UsageError::TotalTokensLimitExceeded { limit });
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum UsageError {
    #[error("request limit exceeded (limit {limit})")]
    RequestLimitExceeded { limit: u64 },
    #[error("tool call limit exceeded (limit {limit})")]
    ToolCallsLimitExceeded { limit: u64 },
    #[error("input token limit exceeded (limit {limit})")]
    InputTokensLimitExceeded { limit: u64 },
    #[error("output token limit exceeded (limit {limit})")]
    OutputTokensLimitExceeded { limit: u64 },
    #[error("total token limit exceeded (limit {limit})")]
    TotalTokensLimitExceeded { limit: u64 },
}
