use crate::audio_block::BlockViewMut;

use super::{AvailableDevices, DeviceConfig, SystemAudioError};

#[cfg(feature = "system-audio-cubeb")]
mod cubeb;
#[cfg(feature = "system-audio-juce")]
mod juce;

pub trait AudioBackend {
    fn new() -> Result<Self, SystemAudioError>
    where
        Self: Sized;

    // Devices
    fn available_devices(&mut self) -> Result<AvailableDevices, SystemAudioError>;
    fn default_config(&mut self) -> Result<DeviceConfig, SystemAudioError>;

    // Audio Stream
    fn start_stream(
        &mut self,
        device_config: &DeviceConfig,
        process_fn: impl FnMut(BlockViewMut<f32>) -> Result<(), &'static str>
            + 'static
            + std::marker::Send
            + std::marker::Sync,
    ) -> Result<(), SystemAudioError>;
    fn stop_stream(&mut self) -> Result<(), SystemAudioError>;
    fn stream_error(&self) -> Result<(), SystemAudioError>;
}
