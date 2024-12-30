use std::sync::Arc;

use neort_blocks::{BlockView, BlockViewMut};
use ringbuf::{
    storage::Heap,
    traits::{Consumer, Observer, Producer, Split},
    wrap::caching::Caching,
    SharedRb,
};

use crate::Float;

pub struct RbProducer<F: Float> {
    producers: Vec<Caching<Arc<SharedRb<Heap<F>>>, true, false>>,
}

impl<F: Float> RbProducer<F> {
    #[rtsan::nonblocking]
    pub fn push_block(&mut self, block: BlockView<F>) -> bool {
        let mut pushed_all = true;
        let num_frames = block.num_frames();
        for (rb, channel) in self.producers.iter_mut().zip(block.channels()) {
            let num_pushed = rb.push_iter(channel.iter().copied());
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

pub struct RbConsumer<F: Float> {
    consumers: Vec<Caching<Arc<SharedRb<Heap<F>>>, false, true>>,
}

impl<F: Float> RbConsumer<F> {
    #[rtsan::nonblocking]
    pub fn pop_block(&mut self, mut block: BlockViewMut<F>) -> bool {
        let mut popped_all = true;
        let num_frames = block.num_frames();
        for (rb, channel) in self.consumers.iter_mut().zip(block.channels_mut()) {
            if rb.occupied_len() < num_frames {
                popped_all = false;
            }
            channel
                .iter_mut()
                .zip(rb.pop_iter())
                .for_each(|(c, r)| *c = r);
        }
        popped_all
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
    use neort_blocks::BlockHeap;

    use super::*;

    #[test]
    fn local_rb() {
        let (mut prod, mut cons) = create_shared_ringbuffer(2, 1024, 0);

        let mut block = BlockHeap::<f32>::new(2, 512);

        block.channel_mut(0).fill(1.0);
        block.channel_mut(1).fill(2.0);

        assert_eq!(cons.num_frames_stored(), 0);
        assert_eq!(cons.num_frames_free(), 1024);

        prod.push_block(block.view());

        assert_eq!(cons.num_frames_stored(), 512);
        assert_eq!(cons.num_frames_free(), 512);
        assert_eq!(
            cons.consumers[0].occupied_len(),
            cons.consumers[1].occupied_len()
        );

        block.channel_mut(0).fill(3.0);
        block.channel_mut(1).fill(4.0);

        prod.push_block(block.view());
        assert_eq!(cons.num_frames_stored(), 1024);
        assert_eq!(cons.num_frames_free(), 0);

        let mut out_block = BlockHeap::new(2, 512);

        let popped_all = cons.pop_block(out_block.view_mut());
        assert!(popped_all);
        assert_eq!(cons.num_frames_stored(), 512);
        assert_eq!(cons.num_frames_free(), 512);

        assert_eq!(out_block.channel(0), &[1.0; 512]);
        assert_eq!(out_block.channel(1), &[2.0; 512]);

        let pushed_all = cons.pop_block(out_block.view_mut());
        assert!(pushed_all);
        assert_eq!(cons.num_frames_stored(), 0);
        assert_eq!(cons.num_frames_free(), 1024);

        assert_eq!(out_block.channel(0), &[3.0; 512]);
        assert_eq!(out_block.channel(1), &[4.0; 512]);
    }
}
