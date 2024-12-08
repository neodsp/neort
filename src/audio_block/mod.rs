use std::ops::Range;

use ndarray::{
    iter::{Lanes, LanesMut},
    ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Dim,
};
use num::Float;

pub use block::Block;
pub use block_view::BlockView;
pub use block_view_mut::BlockViewMut;

mod block;
mod block_view;
mod block_view_mut;

#[derive(Debug, PartialEq, PartialOrd)]
pub enum BufferLayout {
    Sequential,
    Interleaved,
}

/// Trait that is necessary to read from an audio block.
/// All functions inside of this trait are real-time safe
/// and meant to be called inside of your process function.
pub trait BlockRead<F: Float> {
    #[rtsan::nonblocking]
    #[inline(always)]
    fn num_channels(&self) -> u16 {
        self.data().nrows() as u16
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn num_frames(&self) -> usize {
        self.data().ncols()
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn layout(&self) -> BufferLayout {
        if self.data().is_standard_layout() {
            BufferLayout::Sequential
        } else {
            BufferLayout::Interleaved
        }
    }
    #[rtsan::nonblocking]
    #[inline(always)]
    fn sample(&self, ch: u16, frame: usize) -> F {
        self.data()[[ch as usize, frame]]
    }

    fn channel(&self, index: u16) -> ArrayView1<F>;
    fn frame(&self, index: usize) -> ArrayView1<F>;
    fn channels(&self) -> Lanes<F, Dim<[usize; 1]>>;
    fn frames(&self) -> Lanes<F, Dim<[usize; 1]>>;
    fn view(&self) -> BlockView<F>;
    // TODO: write test
    fn view_slice(&self, range: Range<usize>) -> BlockView<F>;
    // TODO: write test
    fn data(&self) -> ArrayView2<F>;
    /// This can return sequential or interleaved data.
    /// The layout can be checked with [`BlockView::layout`] or [`BlockViewMut::layout`].
    fn raw_buffer(&self) -> &[F];
}

/// Trait that is necessary to write to an audio block.
/// All functions inside of this trait are real-time safe
/// and meant to be called inside of your process function.
pub trait BlockWrite<F: Float>: BlockRead<F> {
    fn sample_mut(&mut self, ch: u16, frame: usize) -> &mut F;
    fn channel_mut(&mut self, index: u16) -> ArrayViewMut1<F>;
    fn frame_mut(&mut self, index: usize) -> ArrayViewMut1<F>;
    fn channels_mut(&mut self) -> LanesMut<F, Dim<[usize; 1]>>;
    fn frames_mut(&mut self) -> LanesMut<F, Dim<[usize; 1]>>;

    fn view_mut(&mut self) -> BlockViewMut<F>;

    // TODO: write test
    fn view_slice_mut(&mut self, range: Range<usize>) -> BlockViewMut<F>;

    /// This can return sequential or interleaved data.
    /// The layout can be checked with [`BlockView::layout`] or [`BlockViewMut::layout`].
    fn raw_buffer_mut(&mut self) -> &mut [F];
    // TODO: write test
    fn data_mut(&mut self) -> ArrayViewMut2<F>;

    // TODO: write test
    fn clear(&mut self) {
        self.data_mut().fill(F::zero());
    }
}
