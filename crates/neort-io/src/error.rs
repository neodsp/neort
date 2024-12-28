use thiserror::Error;

#[derive(Debug, Error)]
pub enum SystemAudioError {
    #[error("Unknown Error")]
    UnknownBackendError,
}
