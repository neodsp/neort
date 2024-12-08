mod local;
mod shared;

pub use local::RingbufferLocal;
use num::Float;
pub use shared::RingbufferShared;

use crate::audio_block::{BlockRead, BlockWrite};

pub trait Ringbuffer<F: Float> {
    fn prepare(&mut self, num_channels: u16, frame_capacity: usize, latency: usize);
    fn push_block(&mut self, block: &impl BlockRead<F>) -> bool;
    fn pop_block(&mut self, block: &mut impl BlockWrite<F>) -> bool;
    fn reset(&mut self);

    fn num_frames_stored(&self) -> usize;
    fn num_frames_free(&self) -> usize;
}
