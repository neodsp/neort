use neort_blocks::{BlockView, BlockViewMut};

use crate::{
    available_devices::{AvailableDevices, AvailableSettings},
    device_config::DeviceConfig,
    error::SystemAudioError,
};

#[cfg(feature = "backend-juce")]
pub mod juce_backend;

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
        prepare_fn: impl FnMut(&DeviceConfig) -> Result<(), &'static str> + Send + 'static,
        process_fn: impl FnMut(BlockView<f32>, BlockViewMut<f32>) -> Result<(), &'static str>
            + Send
            + Sync
            + 'static,
    ) -> Result<(), SystemAudioError>;
    fn stop_stream(&mut self) -> Result<(), SystemAudioError>;
    fn stream_error(&self) -> Result<(), SystemAudioError>;
}
