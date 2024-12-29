use neort_blocks::{BlockHeap, BlockView, BlockViewMut};
use num_traits::Float;
use realfft::FftNum;

pub mod base;
pub mod fixed_in;
pub mod fixed_in_out;
pub mod fixed_out;
mod utils;

pub trait Resampler<S: Float + FftNum> {
    #[allow(clippy::result_unit_err)]
    fn process(
        &mut self,
        input: BlockView<S>,
        output: BlockViewMut<S>,
    ) -> Result<(usize, usize), ()>;
    fn input_frames_max(&self) -> usize;
    fn input_frames_next(&self) -> usize;
    fn num_channels(&self) -> usize;
    fn output_frames_max(&self) -> usize;
    fn output_frames_next(&self) -> usize;
    fn output_delay(&self) -> usize;
    fn reset(&mut self);

    fn generate_input_block(&self) -> BlockHeap<S> {
        BlockHeap::new(self.num_channels(), self.input_frames_max())
    }
    fn generate_output_block(&self) -> BlockHeap<S> {
        BlockHeap::new(self.num_channels(), self.output_frames_max())
    }
}
