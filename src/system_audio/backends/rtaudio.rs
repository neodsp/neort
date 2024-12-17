use rtaudio::Host;

use crate::system_audio::SystemAudioError;

use super::AudioBackend;

pub struct RtAudioBackend {
    host: Host,
}

impl AudioBackend for RtAudioBackend {
    fn new() -> Result<Self, crate::system_audio::SystemAudioError>
    where
        Self: Sized,
    {
        let host = Host::new(rtaudio::Api::Unspecified)?;
        Ok(Self { host })
    }

    fn available_devices(
        &mut self,
    ) -> Result<crate::system_audio::AvailableDevices, crate::system_audio::SystemAudioError> {
        todo!()
    }

    fn available_settings(
        &mut self,
        driver: &str,
        input_device: &str,
        output_device: &str,
    ) -> Result<crate::system_audio::AvailableSettings, crate::system_audio::SystemAudioError> {
        todo!()
    }

    fn default_config(
        &mut self,
    ) -> Result<crate::system_audio::DeviceConfig, crate::system_audio::SystemAudioError> {
        todo!()
    }

    fn start_stream(
        &mut self,
        device_config: &crate::system_audio::DeviceConfig,
        process_fn: impl FnMut(crate::audio_block::BlockViewMut<f32>) -> Result<(), &'static str>
            + 'static
            + std::marker::Send
            + std::marker::Sync,
    ) -> Result<(), crate::system_audio::SystemAudioError> {
        todo!()
    }

    fn stop_stream(&mut self) -> Result<(), crate::system_audio::SystemAudioError> {
        todo!()
    }

    fn stream_error(&self) -> Result<(), crate::system_audio::SystemAudioError> {
        todo!()
    }
}

impl From<rtaudio::RtAudioError> for SystemAudioError {
    fn from(value: rtaudio::RtAudioError) -> Self {
        Self::UnknownBackendError(value.to_string())
    }
}
