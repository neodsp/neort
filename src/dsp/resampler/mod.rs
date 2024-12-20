mod base;
mod fixed_in;
mod fixed_in_out;
mod fixed_out;
mod utils;

pub use fixed_in::ResamplerFixedIn;
pub use fixed_in_out::ResamplerFixedInOut;
pub use fixed_out::ResamplerFixedOut;

use crate::{
    audio_block::{Block, BlockRead, BlockWrite},
    Sample,
};

pub trait Resampler<S: Sample> {
    fn process(
        &mut self,
        input: &impl BlockRead<S>,
        output: &mut impl BlockWrite<S>,
    ) -> Result<(usize, usize), ()>;
    fn input_frames_max(&self) -> usize;
    fn input_frames_next(&self) -> usize;
    fn num_channels(&self) -> u16;
    fn output_frames_max(&self) -> usize;
    fn output_frames_next(&self) -> usize;
    fn output_delay(&self) -> usize;
    fn reset(&mut self);

    fn generate_input_block(&self) -> Block<S> {
        Block::new(self.num_channels(), self.input_frames_max())
    }
    fn generate_output_block(&self) -> Block<S> {
        Block::new(self.num_channels(), self.output_frames_max())
    }
}
