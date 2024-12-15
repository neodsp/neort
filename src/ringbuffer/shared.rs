use num::Float;
use ringbuf::{
    storage::Heap,
    traits::{Consumer, Observer, Producer},
    SharedRb,
};

use crate::audio_block::{BlockRead, BlockWrite};

use super::Ringbuffer;

pub struct RingbufferShared<F: Float> {
    ringbuffers: Vec<SharedRb<Heap<F>>>,
}

impl<F: Float> Default for RingbufferShared<F> {
    fn default() -> Self {
        Self {
            ringbuffers: Vec::new(),
        }
    }
}

impl<F: Float> Ringbuffer<F> for RingbufferShared<F> {
    fn prepare(&mut self, num_channels: u16, frame_capacity: usize, latency: usize) {
        assert!(latency < frame_capacity);
        self.ringbuffers = Vec::with_capacity(num_channels as usize);
        for _ in 0..num_channels {
            self.ringbuffers.push(SharedRb::new(frame_capacity));
        }

        for rb in self.ringbuffers.iter_mut() {
            for _ in 0..latency {
                if rb.try_push(F::zero()).is_err() {
                    panic!("Ringbuf should be large enough!");
                }
            }
        }
    }

    #[rtsan::nonblocking]
    fn push_block(&mut self, block: &impl BlockRead<F>) -> bool {
        let mut pushed_all = true;
        let num_frames = block.num_frames();
        for (rb, channel) in self.ringbuffers.iter_mut().zip(block.channels()) {
            let num_pushed = rb.push_iter(channel.iter().take(num_frames).copied());
            if num_pushed != num_frames {
                pushed_all = false;
            }
        }
        pushed_all
    }

    #[rtsan::nonblocking]
    fn pop_block(&mut self, block: &mut impl BlockWrite<F>) -> bool {
        let mut popped_all = true;
        let num_frames = block.num_frames();
        for (rb, mut channel) in self.ringbuffers.iter_mut().zip(block.channels_mut()) {
            if rb.occupied_len() < num_frames {
                popped_all = false;
            }
            channel
                .iter_mut()
                .take(num_frames)
                .zip(rb.pop_iter())
                .for_each(|(c, r)| *c = r);
        }
        popped_all
    }

    fn reset(&mut self) {
        for rb in self.ringbuffers.iter_mut() {
            rb.clear();
        }
    }

    fn num_frames_stored(&self) -> usize {
        self.ringbuffers[0].occupied_len()
    }

    fn num_frames_free(&self) -> usize {
        self.ringbuffers[0].vacant_len()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::aview1;

    use crate::audio_block::Block;

    use super::*;

    #[test]
    fn shared_rb() {
        let mut rb = RingbufferShared::<f32>::default();

        rb.prepare(2, 1024, 0);

        let mut block = Block::new(2, 512);

        block.view_mut().channel_mut(0).fill(1.0);
        block.view_mut().channel_mut(1).fill(2.0);

        assert_eq!(rb.num_frames_stored(), 0);
        assert_eq!(rb.num_frames_free(), 1024);

        rb.push_block(&block.view());

        assert_eq!(rb.num_frames_stored(), 512);
        assert_eq!(rb.num_frames_free(), 512);

        block.view_mut().channel_mut(0).fill(3.0);
        block.view_mut().channel_mut(1).fill(4.0);

        rb.push_block(&block.view());
        assert_eq!(rb.num_frames_stored(), 1024);
        assert_eq!(rb.num_frames_free(), 0);

        let mut out_block = Block::new(2, 512);

        let popped_all = rb.pop_block(&mut out_block.view_mut());
        assert!(popped_all);
        assert_eq!(rb.num_frames_stored(), 512);
        assert_eq!(rb.num_frames_free(), 512);

        assert_eq!(out_block.view().channel(0), aview1(&[1.0; 512]));
        assert_eq!(out_block.view().channel(1), aview1(&[2.0; 512]));

        let pushed_all = rb.pop_block(&mut out_block.view_mut());
        assert!(pushed_all);
        assert_eq!(rb.num_frames_stored(), 0);
        assert_eq!(rb.num_frames_free(), 1024);

        assert_eq!(out_block.view().channel(0), aview1(&[3.0; 512]));
        assert_eq!(out_block.view().channel(1), aview1(&[4.0; 512]));
    }
}
