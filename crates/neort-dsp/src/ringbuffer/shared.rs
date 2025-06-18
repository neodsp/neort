use std::sync::Arc;

use audio_blocks::{AudioBlock, AudioBlockMut};
use neort_float::Float;
use ringbuf::{
    storage::Heap,
    traits::{Consumer, Observer, Producer, Split},
    wrap::caching::Caching,
    SharedRb,
};
use rtsan_standalone::nonblocking;

#[derive(Default)]
pub struct RbProducer<F: Float> {
    producers: Vec<Caching<Arc<SharedRb<Heap<F>>>, true, false>>,
}

unsafe impl<F: Float> Send for RbProducer<F> {}
unsafe impl<F: Float> Sync for RbProducer<F> {}

impl<F: Float> RbProducer<F> {
    #[nonblocking]
    pub fn push_block(&mut self, block: impl AudioBlock<F>) -> bool {
        let mut pushed_all = true;
        let num_frames = block.num_frames();
        for (rb, channel) in self.producers.iter_mut().zip(block.channels()) {
            let num_pushed = rb.push_iter(channel.copied());
            if num_pushed != num_frames {
                pushed_all = false;
            }
        }
        pushed_all
    }

    pub fn num_frames_stored(&self) -> usize {
        self.producers[0].occupied_len()
    }

    pub fn num_frames_free(&self) -> usize {
        self.producers[0].vacant_len()
    }
}

#[derive(Default)]
pub struct RbConsumer<F: Float> {
    consumers: Vec<Caching<Arc<SharedRb<Heap<F>>>, false, true>>,
}

unsafe impl<F: Float> Send for RbConsumer<F> {}
unsafe impl<F: Float> Sync for RbConsumer<F> {}

impl<F: Float> RbConsumer<F> {
    #[nonblocking]
    pub fn pop_block(&mut self, mut block: impl AudioBlockMut<F>) -> bool {
        let mut block_filled = true;
        let num_frames = block.num_frames();
        for (rb, channel) in self.consumers.iter_mut().zip(block.channels_mut()) {
            if rb.occupied_len() < num_frames {
                block_filled = false;
            }
            channel.zip(rb.pop_iter()).for_each(|(a, b)| *a = b);
        }
        block_filled
    }

    pub fn reset(&mut self) {
        for rb in self.consumers.iter_mut() {
            rb.clear();
        }
    }

    pub fn num_frames_stored(&self) -> usize {
        self.consumers[0].occupied_len()
    }

    pub fn num_frames_free(&self) -> usize {
        self.consumers[0].vacant_len()
    }
}

pub fn create_shared_ringbuffer<F: Float>(
    num_channels: usize,
    frame_capacity: usize,
    latency: usize,
) -> (RbProducer<F>, RbConsumer<F>) {
    assert!(latency < frame_capacity);
    let mut producers = Vec::with_capacity(num_channels);
    let mut consumers = Vec::with_capacity(num_channels);
    for _ in 0..num_channels {
        let (prod, cons) = SharedRb::new(frame_capacity).split();
        producers.push(prod);
        consumers.push(cons);
    }

    for rb in producers.iter_mut() {
        for _ in 0..latency {
            if rb.try_push(F::zero()).is_err() {
                panic!("Ringbuf should be large enough!");
            }
        }
    }

    (RbProducer { producers }, RbConsumer { consumers })
}

#[cfg(test)]
mod tests {

    use audio_blocks::Stacked;

    use super::*;

    #[test]
    fn shared_rb() {
        let (mut prod, mut cons) = create_shared_ringbuffer(2, 1024, 0);

        let mut block = Stacked::<f32>::new(2, 512);

        block.channel_slice_mut(0).unwrap().fill(1.0);
        block.channel_slice_mut(1).unwrap().fill(2.0);

        assert_eq!(cons.num_frames_stored(), 0);
        assert_eq!(cons.num_frames_free(), 1024);

        prod.push_block(block.view());

        assert_eq!(cons.num_frames_stored(), 512);
        assert_eq!(cons.num_frames_free(), 512);
        assert_eq!(
            cons.consumers[0].occupied_len(),
            cons.consumers[1].occupied_len()
        );

        block.channel_slice_mut(0).unwrap().fill(3.0);
        block.channel_slice_mut(1).unwrap().fill(4.0);

        prod.push_block(block.view());
        assert_eq!(cons.num_frames_stored(), 1024);
        assert_eq!(cons.num_frames_free(), 0);

        let mut out_block = Stacked::new(2, 512);

        let popped_all = cons.pop_block(out_block.view_mut());
        assert!(popped_all);
        assert_eq!(cons.num_frames_stored(), 512);
        assert_eq!(cons.num_frames_free(), 512);

        assert_eq!(out_block.channel_slice(0).unwrap(), &[1.0; 512]);
        assert_eq!(out_block.channel_slice(1).unwrap(), &[2.0; 512]);

        let pushed_all = cons.pop_block(out_block.view_mut());
        assert!(pushed_all);
        assert_eq!(cons.num_frames_stored(), 0);
        assert_eq!(cons.num_frames_free(), 1024);

        assert_eq!(out_block.channel_slice(0).unwrap(), &[3.0; 512]);
        assert_eq!(out_block.channel_slice(1).unwrap(), &[4.0; 512]);
    }
}
