use std::{marker::PhantomData, sync::atomic::Ordering};

use atomic_float::AtomicF32;

use crate::{
    audio_block::BlockWrite,
    audio_processor::{AudioSettings, Processor},
    Sample,
};

#[allow(unused)]
pub enum GainParameter {
    Gain { ch: u16, gain: f32 },
}

#[derive(Default)]
pub struct Gain<S: Sample> {
    current_gains: Vec<AtomicF32>,
    _phantom: PhantomData<S>,
}

impl<S: Sample> Processor<S> for Gain<S> {
    type Result = ();
    type Parameter = GainParameter;

    fn set_parameter(&mut self, param: Self::Parameter) -> Self::Result {
        match param {
            GainParameter::Gain { ch, gain } => {
                self.current_gains[ch as usize].store(gain, Ordering::Relaxed);
            }
        }
    }

    fn prepare(&mut self, settings: &AudioSettings) -> Self::Result {
        self.current_gains
            .resize_with(settings.num_channels as usize, || AtomicF32::new(1.0));
    }

    #[rtsan::nonblocking]
    fn process(&mut self, audio_block: &mut impl BlockWrite<S>) -> Self::Result {
        for ch in 0..audio_block.num_channels() {
            let gain = S::from(self.current_gains[ch as usize].load(Ordering::Relaxed)).unwrap();
            audio_block.channel_mut(ch).mapv_inplace(|v| v * gain);
        }
    }

    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use ndarray::{array, aview1};

    use crate::audio_block::{Block, BlockRead};

    use super::*;

    #[test]
    fn gain_process() {
        let mut block = Block::from_array(array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);

        let mut gain = Gain::default();

        gain.prepare(&AudioSettings {
            sample_rate: 44100.0,
            num_channels: 2,
            max_num_frames: 3,
        });

        gain.set_parameter(GainParameter::Gain { ch: 0, gain: 2.0 });
        gain.set_parameter(GainParameter::Gain { ch: 1, gain: 4.0 });

        gain.process(&mut block.view_mut());

        assert_eq!(block.channel(0), aview1(&[2.0, 2.0, 2.0]));
        assert_eq!(block.channel(1), aview1(&[4.0, 4.0, 4.0]));
    }
}
