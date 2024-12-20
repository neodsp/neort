mod local;
mod shared;

use crate::Sample;
pub use local::RingbufferLocal;
pub use shared::RingbufferShared;

use crate::audio_block::{BlockRead, BlockWrite};

pub trait Ringbuffer<S: Sample> {
    fn prepare(&mut self, num_channels: u16, frame_capacity: usize, latency: usize);
    fn push_block(&mut self, block: &impl BlockRead<S>) -> bool;
    fn pop_block(&mut self, block: &mut impl BlockWrite<S>) -> bool;
    fn reset(&mut self);

    fn num_frames_stored(&self) -> usize;
    fn num_frames_free(&self) -> usize;
}
