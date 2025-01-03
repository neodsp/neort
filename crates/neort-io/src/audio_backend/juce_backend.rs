use cxx_juce::{
    juce_audio_devices::{
        AudioCallbackHandle, AudioDeviceManager, AudioIODevice, AudioIODeviceCallback,
        AudioIODeviceType, ChannelCount,
    },
    JUCE,
};
use neort_blocks::{BlockHeap, BlockView, BlockViewMut};

use crate::{
    available_devices::{AvailableDevices, AvailableSettings, Driver},
    device_config::DeviceConfig,
    error::SystemAudioError,
};

use super::AudioBackend;

pub struct JuceBackend {
    device_manager: AudioDeviceManager,
    handle: Option<AudioCallbackHandle>,
}

impl AudioBackend for JuceBackend {
    fn new() -> Result<Self, SystemAudioError>
    where
        Self: Sized,
    {
        let mut device_manager = AudioDeviceManager::new(&JUCE::initialise());
        device_manager
            .initialise(256, 256)
            .map_err(|_| SystemAudioError::UnknownBackendError)?;
        Ok(Self {
            device_manager,
            handle: None,
        })
    }

    fn default_config(&mut self) -> Result<DeviceConfig, SystemAudioError> {
        let mut default_device = self
            .device_manager
            .current_device()
            .ok_or(SystemAudioError::UnknownBackendError)?;

        Ok(DeviceConfig {
            driver: default_device.type_name().to_string(),
            input_device: default_device.name().to_string(),
            output_device: default_device.name().to_string(),
            sample_rate: default_device.sample_rate() as usize,
            num_input_ch: default_device.input_channels() as usize,
            num_output_ch: default_device.output_channels() as usize,
            num_frames: default_device.buffer_size(),
        })
    }

    fn available_devices(&mut self) -> Result<AvailableDevices, SystemAudioError> {
        Ok(AvailableDevices {
            drivers: self
                .device_manager
                .device_types()
                .iter_mut()
                .map(|d| {
                    d.scan_for_devices();
                    Driver {
                        name: d.name(),
                        input_devices: d.input_devices(),
                        output_devices: d.output_devices(),
                    }
                })
                .collect(),
        })
    }

    fn available_settings(
        &mut self,
        config: &DeviceConfig,
    ) -> Result<AvailableSettings, SystemAudioError> {
        self.device_manager
            .set_current_audio_device_type(&config.driver);

        self.device_manager.set_audio_device_setup(
            &self
                .device_manager
                .audio_device_setup()
                .with_input_device_name(&config.input_device)
                .with_output_device_name(&config.output_device),
        );

        let mut device = self
            .device_manager
            .current_device()
            .ok_or(SystemAudioError::UnknownBackendError)?;

        Ok(AvailableSettings {
            num_input_ch: device.input_channels() as usize,
            num_output_ch: device.output_channels() as usize,
            sample_rates: device
                .available_sample_rates()
                .iter()
                .map(|&sr| sr as usize)
                .collect(),
            num_frames: device.available_buffer_sizes(),
            default_sample_rate: device.sample_rate() as usize,
            default_num_frames: device.buffer_size(),
        })
    }

    fn start_stream(
        &mut self,
        device_config: &DeviceConfig,
        prepare_fn: impl FnMut(&DeviceConfig) -> Result<(), &'static str> + Send + 'static,
        process_fn: impl FnMut(BlockView<f32>, BlockViewMut<f32>) -> Result<(), &'static str>
            + Send
            + Sync
            + 'static,
    ) -> Result<(), SystemAudioError> {
        self.device_manager
            .set_current_audio_device_type(&device_config.driver);

        let setup = self
            .device_manager
            .audio_device_setup()
            .with_input_device_name(&device_config.input_device)
            .with_output_device_name(&device_config.output_device)
            .with_input_channels(ChannelCount::Custom(device_config.num_input_ch as i32))
            .with_output_channels(ChannelCount::Custom(device_config.num_output_ch as i32))
            .with_sample_rate(device_config.sample_rate as f64)
            .with_buffer_size(device_config.num_frames);
        self.device_manager.set_audio_device_setup(&setup);

        self.handle = Some(
            self.device_manager
                .add_audio_callback(JuceAudioCallback::new(
                    device_config.clone(),
                    prepare_fn,
                    process_fn,
                )),
        );
        Ok(())
    }

    fn stop_stream(&mut self) -> Result<(), SystemAudioError> {
        if let Some(handle) = self.handle.take() {
            self.device_manager.remove_audio_callback(handle);
        }
        Ok(())
    }

    fn stream_error(&self) -> Result<(), SystemAudioError> {
        todo!()
    }
}

#[allow(clippy::complexity)]
pub struct JuceAudioCallback {
    device_config: DeviceConfig,
    prepare_fn: Box<dyn FnMut(&DeviceConfig) -> Result<(), &'static str> + Send + 'static>,
    process_fn: Box<
        dyn FnMut(BlockView<f32>, BlockViewMut<f32>) -> Result<(), &'static str> + Send + 'static,
    >,
    input_block: BlockHeap<f32>,
    output_block: BlockHeap<f32>,
}

impl JuceAudioCallback {
    pub fn new(
        device_config: DeviceConfig,
        prepare: impl FnMut(&DeviceConfig) -> Result<(), &'static str> + Send + 'static,
        process_fn: impl FnMut(BlockView<f32>, BlockViewMut<f32>) -> Result<(), &'static str>
            + Send
            + 'static,
    ) -> Self {
        Self {
            device_config,
            prepare_fn: Box::new(prepare),
            process_fn: Box::new(process_fn),
            input_block: BlockHeap::default(),
            output_block: BlockHeap::default(),
        }
    }
}

impl AudioIODeviceCallback for JuceAudioCallback {
    fn about_to_start(&mut self, device: &mut dyn AudioIODevice) {
        self.input_block = BlockHeap::new(device.input_channels() as usize, device.buffer_size());
        self.output_block = BlockHeap::new(device.output_channels() as usize, device.buffer_size());
        self.device_config.input_device = device.name().to_string();
        self.device_config.output_device = device.name().to_string();
        self.device_config.num_input_ch = device.input_channels() as usize;
        self.device_config.num_output_ch = device.output_channels() as usize;
        self.device_config.sample_rate = device.sample_rate() as usize;
        self.device_config.num_frames = device.buffer_size();
        self.prepare_fn.as_mut()(&self.device_config).unwrap();
    }

    fn process_block(
        &mut self,
        input: &cxx_juce::juce_audio_devices::InputAudioSampleBuffer<'_>,
        output: &mut cxx_juce::juce_audio_devices::OutputAudioSampleBuffer<'_>,
    ) {
        // resize in case less samples are received
        self.input_block.set_num_frames_visible(input.samples());
        self.output_block.set_num_frames_visible(output.samples());

        for ch in 0..input.channels() {
            for frame in 0..input.samples() {
                self.input_block[ch][frame] = input[ch][frame];
            }
        }

        self.process_fn.as_mut()(self.input_block.view(), self.output_block.view_mut()).unwrap();

        for ch in 0..output.channels() {
            for frame in 0..output.samples() {
                output[ch][frame] = self.output_block[ch][frame];
            }
        }
    }

    fn stopped(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[ignore = "manual test"]
    #[test]
    fn test_audio_stream() -> Result<(), SystemAudioError> {
        let mut backend = JuceBackend::new()?;
        let mut config = backend.default_config()?;
        config.num_input_ch = 2;
        config.num_output_ch = 2;
        let _available: Result<AvailableDevices, SystemAudioError> = backend.available_devices();
        let _settings = backend.available_settings(&config)?;
        backend
            .start_stream(
                &config,
                |config| {
                    dbg!(config);
                    Ok(())
                },
                |input, mut output| {
                    output.copy_from_block(&input);
                    Ok(())
                },
            )
            .unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
        backend.stop_stream().unwrap();

        Ok(())
    }
}
