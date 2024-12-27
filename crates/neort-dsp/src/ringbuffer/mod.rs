use neort_blocks::{BlockView, BlockViewMut};

use crate::Float;

pub mod local;
pub mod shared;

pub trait Ringbuffer<F: Float> {
    fn prepare(&mut self, num_channels: usize, frame_capacity: usize, latency: usize);
    fn push_block(&mut self, block: BlockView<F>) -> bool;
    fn pop_block(&mut self, block: BlockViewMut<F>) -> bool;
    fn reset(&mut self);

    fn num_frames_stored(&self) -> usize;
    fn num_frames_free(&self) -> usize;
}
