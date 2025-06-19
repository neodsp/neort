use audio_blocks::AudioBlockMut;
use neort_float::Float;

use crate::ringbuffer::local::RingbufferLocal;

#[derive(Default)]
pub struct Delay<F: Float> {
    rb: RingbufferLocal<F>,
}

impl<F: Float> Delay<F> {
    pub fn prepare(&mut self, num_channels: u16, frame_capacity: usize, latency: usize) {
        self.rb.prepare(num_channels, frame_capacity, latency);
    }

    pub fn process(&mut self, mut block: impl AudioBlockMut<F>) {
        assert!(self.rb.push_block(block.view()));
        assert!(self.rb.pop_block(block.view_mut()))
    }
}
