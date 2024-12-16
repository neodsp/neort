use std::{env, ffi::CString};

use cubeb::{ChannelLayout, Context, DeviceType, Stream};

use crate::{
    audio_block::{Block, BlockRead, BlockWrite},
    system_audio::{AvailableDevices, Driver, InputDevice, OutputDevice, SystemAudioError},
};

use super::AudioBackend;

pub struct MultichannelFrame<T, const N: usize> {
    pub data: [T; N],
}

type Frame = MultichannelFrame<f32, 2>;

pub struct CubebBackend {
    ctx: Context,
    stream: Option<Stream<Frame>>,
}

pub fn init<T: Into<Vec<u8>>>(ctx_name: T) -> Result<Context, cubeb::Error> {
    let backend = match env::var("CUBEB_BACKEND") {
        Ok(s) => Some(s),
        Err(_) => None,
    };

    let ctx_name = CString::new(ctx_name).unwrap();
    let ctx = Context::init(Some(ctx_name.as_c_str()), None);
    if let Ok(ref ctx) = ctx {
        if let Some(ref backend) = backend {
            let ctx_backend = ctx.backend_id();
            if backend != ctx_backend {}
        }
    }

    ctx
}

impl AudioBackend for CubebBackend {
    fn new() -> Result<Self, crate::system_audio::SystemAudioError>
    where
        Self: Sized,
    {
        Ok(Self {
            ctx: init("neort cubeb backend")?,
            stream: None,
        })
    }

    fn available_devices(
        &mut self,
    ) -> Result<crate::system_audio::AvailableDevices, crate::system_audio::SystemAudioError> {
        let input_devices = self
            .ctx
            .enumerate_devices(DeviceType::INPUT)?
            .iter()
            .map(|d| InputDevice {
                name: d.friendly_name().unwrap_or_default().to_string(),
                num_ch: d.max_channels() as u16,
            })
            .collect::<Vec<_>>();

        let output_devices = self
            .ctx
            .enumerate_devices(DeviceType::OUTPUT)?
            .iter()
            .map(|d| OutputDevice {
                name: d.friendly_name().unwrap_or_default().to_string(),
                num_ch: d.max_channels() as u16,
            })
            .collect::<Vec<_>>();

        Ok(AvailableDevices {
            drivers: vec![Driver {
                name: self.ctx.backend_id().to_string(),
                input_devices,
                output_devices,
            }],
        })
    }

    fn default_config(
        &mut self,
    ) -> Result<crate::system_audio::DeviceConfig, crate::system_audio::SystemAudioError> {
        todo!()
    }

    fn start_stream(
        &mut self,
        device_config: &crate::system_audio::DeviceConfig,
        mut process_fn: impl FnMut(crate::audio_block::BlockViewMut<f32>) -> Result<(), &'static str>
            + 'static
            + std::marker::Send
            + std::marker::Sync,
    ) -> Result<(), crate::system_audio::SystemAudioError> {
        let params = cubeb::StreamParamsBuilder::new()
            .format(cubeb::SampleFormat::Float32LE)
            .rate(device_config.sample_rate)
            .channels(2)
            .layout(ChannelLayout::UNDEFINED)
            .take();

        let mut block = Block::new(2, 10000);
        dbg!("moop");

        let mut builder = cubeb::StreamBuilder::new();
        builder
            .name("neort process")
            .default_input(&params)
            .default_output(&params)
            .latency(device_config.num_frames as u32)
            .data_callback(move |input: &[Frame], output: &mut [Frame]| {
                // dbg!(input.len(), output.len());
                block.set_num_frames(input.len());
                let mut block = block.view_mut();
                for (input, mut block) in input.iter().zip(block.frames_mut().into_iter()) {
                    for (in_sample, block_sample) in input.data.iter().zip(block.iter_mut()) {
                        *block_sample = *in_sample;
                    }
                }

                process_fn(block.view_mut()).unwrap();

                for (output, block) in output.iter_mut().zip(block.frames().into_iter()) {
                    for (out_sample, block_sample) in output.data.iter_mut().zip(block.iter()) {
                        *out_sample = *block_sample;
                    }
                }
                output.len() as isize
            })
            .state_callback(|state| {
                println!("stream {:?}", state);
            });

        println!("meep");

        let stream = builder.init(&self.ctx)?;

        println!("first");

        stream.start()?;

        println!("hello");
        println!(
            "{}, {}",
            stream.latency().unwrap(),
            stream.input_latency().unwrap()
        );

        self.stream = Some(stream);

        Ok(())
    }

    fn stop_stream(&mut self) -> Result<(), crate::system_audio::SystemAudioError> {
        if let Some(stream) = self.stream.as_mut() {
            stream.stop()?;
        }
        Ok(())
    }

    fn stream_error(&self) -> Result<(), crate::system_audio::SystemAudioError> {
        todo!()
    }
}

impl From<cubeb::Error> for SystemAudioError {
    fn from(value: cubeb::Error) -> Self {
        Self::UnknownBackendError(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use crate::system_audio::DeviceConfig;

    use super::*;

    #[test]
    fn test() -> Result<(), SystemAudioError> {
        let mut backend = CubebBackend::new()?;
        let devices = backend.available_devices()?;
        backend.start_stream(
            &DeviceConfig {
                driver: "rust-pulse".to_string(),
                input_device: String::new(),
                output_device: String::new(),
                sample_rate: 48000,
                num_input_ch: 2,
                num_output_ch: 2,
                num_frames: 512,
            },
            |block| Ok(()),
        )?;

        std::thread::sleep(std::time::Duration::from_secs(10));

        backend.stop_stream()?;

        Ok(())
    }
}
