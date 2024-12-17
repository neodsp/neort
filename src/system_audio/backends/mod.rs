use crate::audio_block::BlockViewMut;

use super::{
    device_config::ProcessConfig, AvailableDevices, AvailableSettings, DeviceConfig,
    SystemAudioError,
};

#[cfg(feature = "system-audio-cubeb")]
mod cubeb;
#[cfg(feature = "system-audio-juce")]
mod juce;
#[cfg(feature = "system-audio-rtaudio")]
mod rtaudio;

pub trait AudioBackend {
    fn new() -> Result<Self, SystemAudioError>
    where
        Self: Sized;

    // Devices
    fn default_config(&mut self) -> Result<DeviceConfig, SystemAudioError>;
    fn available_devices(&mut self) -> Result<AvailableDevices, SystemAudioError>;
    fn available_settings(
        &mut self,
        config: &DeviceConfig,
    ) -> Result<AvailableSettings, SystemAudioError>;

    // Audio Stream
    fn start_stream(
        &mut self,
        device_config: &DeviceConfig,
        prepare_fn: impl FnMut(ProcessConfig) -> Result<(), &'static str> + Send + 'static,
        process_fn: impl FnMut(BlockViewMut<f32>) -> Result<(), &'static str> + Send + Sync + 'static,
    ) -> Result<(), SystemAudioError>;
    fn stop_stream(&mut self) -> Result<(), SystemAudioError>;
    fn stream_error(&self) -> Result<(), SystemAudioError>;
}
