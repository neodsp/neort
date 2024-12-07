mod base;
mod fixed_in;
mod fixed_in_out;
mod fixed_out;
mod utils;

pub use fixed_in::ResamplerFixedIn;
pub use fixed_in_out::ResamplerFixedInOut;
pub use fixed_out::ResamplerFixedOut;
use num::Float;
use realfft::FftNum;

use crate::audio_block::{Block, BlockRead, BlockWrite};

pub trait Resampler<F: Float + FftNum> {
    fn process(
        &mut self,
        input: &impl BlockRead<F>,
        output: &mut impl BlockWrite<F>,
    ) -> Result<(usize, usize), ()>;
    fn input_frames_max(&self) -> usize;
    fn input_frames_next(&self) -> usize;
    fn num_channels(&self) -> u16;
    fn output_frames_max(&self) -> usize;
    fn output_frames_next(&self) -> usize;
    fn output_delay(&self) -> usize;
    fn reset(&mut self);

    fn generate_input_block(&self) -> Block<F> {
        Block::new(self.num_channels(), self.input_frames_max())
    }
    fn generate_output_block(&self) -> Block<F> {
        Block::new(self.num_channels(), self.output_frames_max())
    }
}
