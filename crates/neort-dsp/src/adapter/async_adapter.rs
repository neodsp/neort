use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use super::resamplers::Resamplers;
use neort_blocks::{BlockHeap, BlockViewMut};
use neort_float::Float;

use crate::ringbuffer::shared::{create_shared_ringbuffer, RbConsumer, RbProducer};

#[derive(Default)]
pub struct AsyncAdapter<F: Float> {
    input_prod: RbProducer<F>,
    output_cons: RbConsumer<F>,
    handle: Option<JoinHandle<()>>,
    terminate_flag: Arc<AtomicBool>,
}

impl<F: Float> AsyncAdapter<F> {
    #[allow(clippy::new_without_default)]
    pub fn prepare(
        &mut self,
        num_channels: usize,
        system_sample_rate: usize,
        system_max_num_frames: usize,
        user_sample_rate: usize,
        user_num_frames: usize,
        mut process_fn: impl FnMut(BlockViewMut<F>) + Send + 'static,
    ) {
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
        self.input_prod = input_prod;
        self.output_cons = output_cons;

        // stop old thread
        if let Some(handle) = self.handle.take() {
            self.terminate_flag.store(true, Ordering::Relaxed);
            handle.join().unwrap();
        }

        self.terminate_flag = Arc::new(AtomicBool::new(false));
        let terminate_flag_clone = Arc::clone(&self.terminate_flag);

        self.handle = Some(std::thread::spawn(move || {
            while !terminate_flag_clone.load(Ordering::Relaxed) {
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
                        assert!(input_cons.pop_block(process_block.view_mut()));

                        process_fn(process_block.view_mut());

                        assert!(output_prod.push_block(process_block.view()));
                    }
                }
            }
        }));
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

impl<F: Float> Drop for AsyncAdapter<F> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.terminate_flag.store(true, Ordering::Relaxed);
            handle.join().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // const CALLED_FLAG: AtomicBool = AtomicBool::new(false);

    #[test]
    fn test_name() {
        let called_flag = Arc::new(AtomicBool::new(false));
        let called_flag_clone = Arc::clone(&called_flag);

        let user_num_frames = 64;
        {
            let mut adapter = AsyncAdapter::<f32>::default();

            adapter.prepare(2, 48000, 1024, 16000, user_num_frames, move |block| {
                assert_eq!(block.num_frames(), user_num_frames);
                called_flag_clone.store(true, Ordering::Relaxed);
                dbg!("I've been called!");
            });

            let mut block = BlockHeap::new(2, 256);

            adapter.process(block.view_mut());
            adapter.process(block.view_mut());
            adapter.process(block.view_mut());
            adapter.process(block.view_mut());
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        assert!(called_flag.load(Ordering::Relaxed));
    }
}
