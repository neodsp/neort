use atomic_float::AtomicF32;
use num::Float;

use crate::{
    audio_block::{Block, BlockWrite},
    audio_processor::{AudioSettings, Processor},
};

use super::resampler::{lanczos_resampler, sinc_resampler::resample};

pub struct AdapterConfig {
    wanted_sample_rate: f64,
    wanted_num_frames: u32,
}

pub struct Adapter<F: Float> {
    user_block: Block<F>,
    wanted_config: AdapterConfig,
}

impl<F: Float> Processor<F> for Adapter<F> {
    type Result = ();

    type Parameter = AdapterConfig;

    /// The adapter-config will only be updated when the next prepare is called
    fn set_parameter(&mut self, param: Self::Parameter) -> Self::Result {
        self.wanted_config = param;
    }

    fn prepare(&mut self, settings: &AudioSettings) -> Self::Result {
        self.user_block = lanczos_resampler::generate_output_block(
            settings.sample_rate,
            settings.max_num_frames,
            self.wanted_config.wanted_sample_rate,
        );
    }

    fn process(&mut self, block: &mut impl BlockWrite<F>) -> Self::Result {
        let frames_written_input =
            lanczos_resampler::process(block, block.num_frames(), &mut self.user_block);

        let frames_written_output =
            lanczos_resampler::process(&self.user_block, frames_written_input, block);
    }

    fn reset(&mut self) {
        todo!()
    }
}
