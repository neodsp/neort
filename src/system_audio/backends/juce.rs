use cxx_juce::{
    juce_audio_devices::{
        AudioCallbackHandle, AudioDeviceManager, AudioIODevice, AudioIODeviceCallback,
        AudioIODeviceType,
    },
    JUCE,
};
use lazy_static::lazy_static;

use crate::{
    audio_block::{Block, BlockRead, BlockWrite},
    system_audio::{
        AvailableDevices, DeviceConfig, Driver, InputDevice, OutputDevice, SystemAudioError,
    },
};

use super::AudioBackend;

lazy_static! {
    static ref JUCE_GLOBAL: JUCE<'static> = JUCE::initialise();
}

pub struct JuceBackend<'a> {
    device_manager: AudioDeviceManager<'a>,
    handle: Option<AudioCallbackHandle>,
}

impl<'a> AudioBackend for JuceBackend<'a> {
    fn new() -> Result<Self, crate::system_audio::SystemAudioError>
    where
        Self: Sized,
    {
        let mut device_manager = AudioDeviceManager::new(&JUCE_GLOBAL);
        device_manager.initialise(256, 256).map_err(|_| {
            SystemAudioError::UnknownBackendError("Could not Initialize".to_string())
        })?;
        Ok(Self {
            device_manager,
            handle: None,
        })
    }

    fn available_devices(
        &mut self,
    ) -> Result<crate::system_audio::AvailableDevices, crate::system_audio::SystemAudioError> {
        Ok(AvailableDevices {
            drivers: self
                .device_manager
                .device_types()
                .iter_mut()
                .map(|d| {
                    d.scan_for_devices();
                    Driver {
                        name: d.name(),
                        input_devices: d
                            .input_devices()
                            .iter()
                            .map(|d| InputDevice {
                                name: d.clone(),
                                num_ch: 256,
                            })
                            .collect(),
                        output_devices: d
                            .output_devices()
                            .iter()
                            .map(|d| OutputDevice {
                                name: d.clone(),
                                num_ch: 256,
                            })
                            .collect(),
                    }
                })
                .collect(),
        })
    }

    fn default_config(
        &mut self,
    ) -> Result<crate::system_audio::DeviceConfig, crate::system_audio::SystemAudioError> {
        let mut default_device = self
            .device_manager
            .current_device()
            .ok_or(SystemAudioError::DeviceNotFound(String::from("Driver")))?;

        Ok(DeviceConfig {
            driver: default_device.type_name().to_string(),
            input_device: default_device.name().to_string(),
            output_device: default_device.name().to_string(),
            sample_rate: default_device.sample_rate() as u32,
            num_input_ch: default_device.input_channels() as u16,
            num_output_ch: default_device.output_channels() as u16,
            num_frames: default_device.buffer_size(),
        })
    }

    fn start_stream(
        &mut self,
        device_config: &crate::system_audio::DeviceConfig,
        process_fn: impl FnMut(crate::audio_block::BlockViewMut<f32>) -> Result<(), &'static str>
            + 'static
            + std::marker::Send
            + std::marker::Sync,
    ) -> Result<(), crate::system_audio::SystemAudioError> {
        self.handle = Some(
            self.device_manager
                .add_audio_callback(JuceAudioCallback::new(process_fn)),
        );
        Ok(())
    }

    fn stop_stream(&mut self) -> Result<(), crate::system_audio::SystemAudioError> {
        if let Some(handle) = self.handle.take() {
            self.device_manager.remove_audio_callback(handle);
        }
        Ok(())
    }

    fn stream_error(&self) -> Result<(), crate::system_audio::SystemAudioError> {
        todo!()
    }
}

pub struct JuceAudioCallback {
    process_fn: Box<
        dyn FnMut(crate::audio_block::BlockViewMut<f32>) -> Result<(), &'static str>
            + 'static
            + Send,
    >,
    block: Block<f32>,
}

impl JuceAudioCallback {
    pub fn new(
        process_fn: impl FnMut(crate::audio_block::BlockViewMut<f32>) -> Result<(), &'static str>
            + 'static
            + Send,
    ) -> Self {
        Self {
            process_fn: Box::new(process_fn),
            block: Block::default(),
        }
    }
}

impl AudioIODeviceCallback for JuceAudioCallback {
    fn about_to_start(&mut self, device: &mut dyn AudioIODevice) {
        dbg!(device.name());
        dbg!(device.buffer_size());
        dbg!(device.sample_rate());
        dbg!(device.input_channels());
        dbg!(device.output_channels());
        let num_channels = device.input_channels().min(device.output_channels());
        self.block = Block::new(num_channels as u16, device.buffer_size());
    }

    fn process_block(
        &mut self,
        input: &cxx_juce::juce_audio_devices::InputAudioSampleBuffer<'_>,
        output: &mut cxx_juce::juce_audio_devices::OutputAudioSampleBuffer<'_>,
    ) {
        let mut block_view = self.block.view_mut();

        for (i, mut ch) in block_view.channels_mut().into_iter().enumerate() {
            for (j, sample) in ch.iter_mut().enumerate() {
                *sample = input[i][j];
            }
        }

        self.process_fn.as_mut()(block_view.view_mut()).unwrap();

        for (i, ch) in block_view.channels().into_iter().enumerate() {
            for (j, sample) in ch.iter().enumerate() {
                output[i][j] = *sample;
            }
        }
    }

    fn stopped(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() -> Result<(), SystemAudioError> {
        let mut juce = JuceBackend::new()?;
        let config = juce.default_config()?;
        juce.start_stream(&config, |_| Ok(())).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(10));
        Ok(())
    }
}
