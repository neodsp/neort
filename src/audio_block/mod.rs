use ndarray::{ArrayView1, ArrayViewMut1};

pub use block::Block;
pub use block_view::BlockView;
pub use block_view_mut::BlockViewMut;
pub use block_view_stacked::BlockViewStacked;
pub use block_view_stacked_mut::BlockViewStackedMut;
use rtsan::nonblocking;

use crate::Sample;

mod block;
mod block_view;
mod block_view_mut;
mod block_view_stacked;
mod block_view_stacked_mut;

pub enum BufferLayout {
    Interleaved,
    Sequential,
}

/// Trait that is necessary to read from an audio block.
/// All functions inside of this trait are real-time safe
/// and meant to be called inside of your process function.
pub trait BlockRead<S: Sample> {
    fn num_channels(&self) -> u16;
    fn num_frames(&self) -> usize;
    fn sample(&self, ch: u16, frame: usize) -> S;
    fn channel(&self, index: u16) -> ArrayView1<S>;
    fn channels(&self) -> impl Iterator<Item = ndarray::ArrayView1<S>>;
}

/// Trait that is necessary to write to an audio block.
/// All functions inside of this trait are real-time safe
/// and meant to be called inside of your process function.
pub trait BlockWrite<S: Sample>: BlockRead<S> {
    fn sample_mut(&mut self, ch: u16, frame: usize) -> &mut S;
    fn channel_mut(&mut self, index: u16) -> ArrayViewMut1<S>;
    fn channels_mut(&mut self) -> impl Iterator<Item = ndarray::ArrayViewMut1<S>>;
    #[nonblocking]
    fn clear(&mut self) {
        for ch in 0..self.num_channels() {
            self.channel_mut(ch).fill(S::zero());
        }
    }
}
