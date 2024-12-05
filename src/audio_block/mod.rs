use ndarray::{
    iter::{Lanes, LanesMut},
    ArrayView1, ArrayViewMut1, Dim,
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
    fn sample_rate(&self) -> f64;
    fn num_channels(&self) -> u16;
    fn num_frames(&self) -> u32;
    fn layout(&self) -> BufferLayout;

    fn channel(&self, index: u16) -> ArrayView1<F>;
    fn frame(&self, index: u32) -> ArrayView1<F>;
    fn channels(&self) -> Lanes<F, Dim<[usize; 1]>>;
    fn frames(&self) -> Lanes<F, Dim<[usize; 1]>>;

    fn view(&self) -> BlockView<F>;

    /// If the block is a view, this can return sequential or interleaved data.
    /// The layout can be checked with [`BlockView::layout`] or [`BlockViewMut::layout`].
    fn raw_buffer(&self) -> &[F];
}

/// Trait that is necessary to write to an audio block.
/// All functions inside of this trait are real-time safe
/// and meant to be called inside of your process function.
pub trait BlockWrite<F: Float>: BlockRead<F> {
    fn channel_mut(&mut self, index: u16) -> ArrayViewMut1<F>;
    fn frame_mut(&mut self, index: u32) -> ArrayViewMut1<F>;
    fn channels_mut(&mut self) -> LanesMut<F, Dim<[usize; 1]>>;
    fn frames_mut(&mut self) -> LanesMut<F, Dim<[usize; 1]>>;

    fn view_mut(&mut self) -> BlockViewMut<F>;

    /// If the block is a view, this can return sequential or interleaved data.
    /// The layout can be checked with [`BlockView::layout`] or [`BlockViewMut::layout`].
    fn raw_buffer_mut(&mut self) -> &mut [F];
}
