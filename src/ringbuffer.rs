use num::Float;
use ringbuf::{
    storage::Heap,
    traits::{Consumer, Observer, Producer},
    LocalRb,
};

use crate::audio_block::{BlockRead, BlockWrite};

#[derive(Default)]
pub struct Ringbuffer<F: Float> {
    ringbuffers: Vec<LocalRb<Heap<F>>>,
}

impl<F: Float> Ringbuffer<F> {
    pub fn prepare(&mut self, num_channels: u16, frame_capacity: usize, latency: usize) {
        assert!(latency < frame_capacity);
        self.ringbuffers = Vec::with_capacity(num_channels as usize);
        for _ in 0..num_channels {
            self.ringbuffers.push(LocalRb::new(frame_capacity));
        }

        for rb in self.ringbuffers.iter_mut() {
            for _ in 0..latency {
                if rb.try_push(F::zero()).is_err() {
                    panic!("Ringbuf should be large enough!");
                }
            }
        }
    }

    pub fn num_stored(&self) -> usize {
        self.ringbuffers[0].occupied_len()
    }

    pub fn num_free(&self) -> usize {
        self.ringbuffers[0].vacant_len()
    }

    #[rtsan::nonblocking]
    pub fn push(&mut self, block: &impl BlockRead<F>, num_frames: u32) -> bool {
        assert!(num_frames <= block.num_frames());
        let mut pushed_all = true;
        for (rb, channel) in self.ringbuffers.iter_mut().zip(block.channels()) {
            let num_pushed = rb.push_iter(channel.iter().take(num_frames as usize).copied());
            if num_pushed != num_frames as usize {
                pushed_all = false;
            }
        }
        pushed_all
    }

    #[rtsan::nonblocking]
    pub fn pop(&mut self, block: &mut impl BlockWrite<F>, num_frames: u32) -> bool {
        assert!(num_frames <= block.num_frames());
        let mut pushed_all = true;
        for (rb, mut channel) in self.ringbuffers.iter_mut().zip(block.channels_mut()) {
            if rb.occupied_len() < num_frames as usize {
                pushed_all = false;
            }
            channel
                .iter_mut()
                .take(num_frames as usize)
                .zip(rb.pop_iter())
                .for_each(|(c, r)| *c = r);
        }
        pushed_all
    }
}

#[cfg(test)]
mod tests {
    use ndarray::aview1;

    use crate::audio_block::Block;

    use super::*;

    #[test]
    fn test_name() {
        let mut rb = Ringbuffer::<f32>::default();

        rb.prepare(2, 1024, 0);

        let mut block = Block::new(44100.0, 2, 512);

        block.channel_mut(0).fill(1.0);
        block.channel_mut(1).fill(2.0);

        rb.push(&block, block.num_frames());

        block.channel_mut(0).fill(3.0);
        block.channel_mut(1).fill(4.0);

        rb.push(&block, block.num_frames());

        let mut out_block = Block::new(44100.0, 2, 512);

        let popped_all = rb.pop(&mut out_block, 512);
        assert!(popped_all);

        assert_eq!(out_block.channel(0), aview1(&[1.0; 512]));
        assert_eq!(out_block.channel(1), aview1(&[2.0; 512]));

        let pushed_all = rb.pop(&mut out_block, 512);
        assert!(pushed_all);

        assert_eq!(out_block.channel(0), aview1(&[3.0; 512]));
        assert_eq!(out_block.channel(1), aview1(&[4.0; 512]));
    }
}
