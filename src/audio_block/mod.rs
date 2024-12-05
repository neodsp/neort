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

pub enum BufferLayout {
    Sequential,
    Interleaved,
}

pub trait BlockRead<Sample: Float> {
    fn sample_rate(&self) -> f64;
    fn num_channels(&self) -> u16;
    fn num_frames(&self) -> u32;

    fn channel(&self, index: u16) -> ArrayView1<Sample>;
    fn frame(&self, index: u32) -> ArrayView1<Sample>;
    fn channels(&self) -> Lanes<Sample, Dim<[usize; 1]>>;
    fn frames(&self) -> Lanes<Sample, Dim<[usize; 1]>>;

    fn view(&self) -> BlockView<Sample>;

    fn raw_buffer(&self) -> &[Sample];
}

pub trait BlockWrite<Sample: Float>: BlockRead<Sample> {
    fn channel_mut(&mut self, index: u16) -> ArrayViewMut1<Sample>;
    fn frame_mut(&mut self, index: u32) -> ArrayViewMut1<Sample>;
    fn channels_mut(&mut self) -> LanesMut<Sample, Dim<[usize; 1]>>;
    fn frames_mut(&mut self) -> LanesMut<Sample, Dim<[usize; 1]>>;

    fn view_mut(&mut self) -> BlockViewMut<Sample>;

    fn raw_buffer_mut(&mut self) -> &mut [Sample];
}
