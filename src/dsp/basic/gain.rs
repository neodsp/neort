use std::{marker::PhantomData, sync::atomic::Ordering};

use atomic_float::AtomicF32;
use num::Float;

use crate::{
    audio_block::BlockWrite,
    audio_processor::{AudioSettings, Processor},
};

#[allow(unused)]
pub enum GainParameter {
    Gain { ch: u16, gain: f32 },
}

#[derive(Default)]
pub struct Gain<F: Float> {
    current_gains: Vec<AtomicF32>,
    _phantom: PhantomData<F>,
}

impl<F: Float> Processor<F> for Gain<F> {
    type PrepareResult = ();

    type ProcessResult = ();

    type SetParameterResult = ();

    type Parameter = GainParameter;

    fn prepare(&mut self, settings: &AudioSettings) -> Self::PrepareResult {
        self.current_gains
            .resize_with(settings.num_channels as usize, || AtomicF32::new(1.0));
    }

    #[rtsan::nonblocking]
    fn process(&mut self, block: &mut impl BlockWrite<F>) -> Self::ProcessResult {
        for (ch, mut channel) in block.channels_mut().into_iter().enumerate() {
            let gain = F::from(self.current_gains[ch].load(Ordering::Relaxed)).unwrap();
            channel.mapv_inplace(|v| v * gain);
        }
    }

    fn reset(&mut self) {}

    fn set_parameter(&mut self, param: Self::Parameter) -> Self::SetParameterResult {
        match param {
            GainParameter::Gain { ch, gain } => {
                self.current_gains[ch as usize].store(gain, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, aview1};

    use crate::audio_block::{Block, BlockRead};

    use super::*;

    #[test]
    fn gain_process() {
        let mut block = Block::from_array(array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], 44100.0);

        let mut gain = Gain::default();

        gain.prepare(&AudioSettings {
            sample_rate: 44100.0,
            num_channels: 2,
            max_num_frames: 3,
        });

        gain.set_parameter(GainParameter::Gain { ch: 0, gain: 2.0 });
        gain.set_parameter(GainParameter::Gain { ch: 1, gain: 4.0 });

        gain.process(&mut block);

        assert_eq!(block.channel(0), aview1(&[2.0, 2.0, 2.0]));
        assert_eq!(block.channel(1), aview1(&[4.0, 4.0, 4.0]));
    }
}
