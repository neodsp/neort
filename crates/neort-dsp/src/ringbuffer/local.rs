use audio_blocks::{AudioBlock, AudioBlockMut};
use neort_float::Float;
use ringbuf::{
    storage::Heap,
    traits::{Consumer, Observer, Producer},
    LocalRb,
};
use rtsan_standalone::nonblocking;

#[derive(Default)]
pub struct RingbufferLocal<F: Float> {
    ringbuffers: Vec<LocalRb<Heap<F>>>,
    latency: usize,
}

impl<F: Float> RingbufferLocal<F> {
    pub fn prepare(&mut self, num_channels: u16, frame_capacity: usize, latency: usize) {
        assert!(latency < frame_capacity);
        self.ringbuffers = Vec::with_capacity(num_channels as usize);
        for _ in 0..num_channels {
            self.ringbuffers.push(LocalRb::new(frame_capacity));
        }

        self.latency = latency;

        self.apply_latency();
    }

    #[nonblocking]
    pub fn push_block(&mut self, block: impl AudioBlock<F>) -> bool {
        let mut pushed_all = true;
        let num_frames = block.num_frames();
        for (rb, channel) in self.ringbuffers.iter_mut().zip(block.channels()) {
            let num_pushed = rb.push_iter(channel.copied());
            if num_pushed != num_frames {
                pushed_all = false;
            }
        }
        pushed_all
    }

    #[nonblocking]
    pub fn pop_block(&mut self, mut block: impl AudioBlockMut<F>) -> bool {
        let mut popped_all = true;
        let num_frames = block.num_frames();
        for (rb, channel) in self.ringbuffers.iter_mut().zip(block.channels_mut()) {
            if rb.occupied_len() < num_frames {
                popped_all = false;
            }
            channel.zip(rb.pop_iter()).for_each(|(c, r)| *c = r);
        }
        popped_all
    }

    pub fn reset(&mut self) {
        for rb in self.ringbuffers.iter_mut() {
            rb.clear();
        }
        self.apply_latency();
    }

    pub fn num_frames_stored(&self) -> usize {
        self.ringbuffers[0].occupied_len()
    }

    pub fn num_frames_free(&self) -> usize {
        self.ringbuffers[0].vacant_len()
    }

    fn apply_latency(&mut self) {
        for rb in self.ringbuffers.iter_mut() {
            for _ in 0..self.latency {
                if rb.try_push(F::zero()).is_err() {
                    panic!("Ringbuf should be large enough!");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use audio_blocks::Stacked;

    use super::*;

    #[test]
    fn local_rb() {
        let mut rb = RingbufferLocal::<f32>::default();

        rb.prepare(2, 1024, 0);

        let mut block = Stacked::<f32>::new(2, 512);

        block.channel_slice_mut(0).unwrap().fill(1.0);
        block.channel_slice_mut(1).unwrap().fill(2.0);

        assert_eq!(rb.num_frames_stored(), 0);
        assert_eq!(rb.num_frames_free(), 1024);

        rb.push_block(block.view());

        assert_eq!(rb.num_frames_stored(), 512);
        assert_eq!(rb.num_frames_free(), 512);
        assert_eq!(
            rb.ringbuffers[0].occupied_len(),
            rb.ringbuffers[1].occupied_len()
        );

        block.channel_slice_mut(0).unwrap().fill(3.0);
        block.channel_slice_mut(1).unwrap().fill(4.0);

        rb.push_block(block.view());
        assert_eq!(rb.num_frames_stored(), 1024);
        assert_eq!(rb.num_frames_free(), 0);

        let mut out_block = Stacked::new(2, 512);

        let popped_all = rb.pop_block(out_block.view_mut());
        assert!(popped_all);
        assert_eq!(rb.num_frames_stored(), 512);
        assert_eq!(rb.num_frames_free(), 512);

        assert_eq!(out_block.channel_slice(0).unwrap(), &[1.0; 512]);
        assert_eq!(out_block.channel_slice(1).unwrap(), &[2.0; 512]);

        let pushed_all = rb.pop_block(out_block.view_mut());
        assert!(pushed_all);
        assert_eq!(rb.num_frames_stored(), 0);
        assert_eq!(rb.num_frames_free(), 1024);

        assert_eq!(out_block.channel_slice(0).unwrap(), &[3.0; 512]);
        assert_eq!(out_block.channel_slice(1).unwrap(), &[4.0; 512]);
    }
}
