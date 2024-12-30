use super::resamplers::Resamplers;
use neort_blocks::{BlockHeap, BlockViewMut};

use crate::ringbuffer::shared::{create_shared_ringbuffer, RbConsumer, RbProducer};
use crate::Float;

pub struct AsyncAdapter<F: Float> {
    input_prod: RbProducer<F>,
    output_cons: RbConsumer<F>,
}

impl<F: Float> AsyncAdapter<F> {
    #[allow(clippy::new_without_default)]
    pub fn new(
        num_channels: usize,
        system_sample_rate: usize,
        system_max_num_frames: usize,
        user_sample_rate: usize,
        user_num_frames: usize,
        mut process_fn: impl FnMut(BlockViewMut<F>) + Send + 'static,
    ) -> Self {
        let mut resamplers = None;

        if system_sample_rate != user_sample_rate {
            resamplers = Some(Resamplers::<F>::new(
                num_channels,
                system_sample_rate,
                user_sample_rate,
                user_num_frames,
            ));
        }

        let mut process_block = BlockHeap::<F>::new(num_channels, user_num_frames);

        let max_frames = resamplers.as_ref().map(|r| r.frames_max()).unwrap_or(0);

        let max_frames = max_frames.max(system_max_num_frames).max(user_num_frames);

        let (input_prod, mut input_cons) =
            create_shared_ringbuffer(num_channels, max_frames * 3, 0);
        let (mut output_prod, output_cons) =
            create_shared_ringbuffer(num_channels, max_frames * 3, 0);

        std::thread::spawn(move || loop {
            if let Some(resamplers) = resamplers.as_mut() {
                // Resampling necessary
                while input_cons.num_frames_stored() >= resamplers.input_frames_next() {
                    assert!(input_cons.pop_block(resamplers.input_block()));
                    resamplers.process_input(process_block.view_mut());

                    process_fn(process_block.view_mut());

                    resamplers.process_output(process_block.view());
                    assert!(output_prod.push_block(resamplers.output_block()));
                }
            } else {
                // Resampling unnecessary
                while input_cons.num_frames_stored() >= user_num_frames {
                    input_cons.pop_block(process_block.view_mut());

                    process_fn(process_block.view_mut());

                    assert!(output_prod.push_block(process_block.view()));
                }
            }
        });

        Self {
            input_prod,
            output_cons,
        }
    }

    pub fn process(&mut self, mut block: BlockViewMut<F>) {
        assert!(self.input_prod.push_block(block.view()));

        if self.output_cons.num_frames_stored() >= block.num_frames() {
            // go to the latest buffer to reduce latency
            while self.output_cons.num_frames_stored() >= block.num_frames() {
                assert!(self.output_cons.pop_block(block.view_mut()));
            }
        } else {
            block.clear();
        }
    }
}
