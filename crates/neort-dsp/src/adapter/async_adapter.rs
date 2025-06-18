use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use super::resamplers::Resamplers;
use audio_blocks::{AudioBlock, AudioBlockMut, Ops, Stacked};
use event_listener::{Event, Listener};
use neort_float::Float;
use rtsan_standalone::nonblocking;

use crate::ringbuffer::shared::{create_shared_ringbuffer, RbConsumer, RbProducer};

#[derive(Default)]
pub struct AsyncAdapter<F: Float> {
    input_prod: RbProducer<F>,
    output_cons: RbConsumer<F>,
    handle: Option<JoinHandle<()>>,
    event: Arc<Event>,
    terminate_flag: Arc<AtomicBool>,
}

impl<F: Float> AsyncAdapter<F> {
    pub fn prepare(
        &mut self,
        num_channels: u16,
        system_sample_rate: usize,
        system_max_num_frames: usize,
        user_sample_rate: usize,
        user_num_frames: usize,
        mut process_fn: impl FnMut(&mut Stacked<F>) + Send + 'static,
    ) {
        let mut resamplers = None;

        if system_sample_rate != user_sample_rate {
            resamplers = Some(Resamplers::<F>::new(
                num_channels as usize,
                system_sample_rate,
                user_sample_rate,
                user_num_frames,
            ));
        }

        let mut process_block = Stacked::<F>::new(num_channels, user_num_frames);

        let max_frames = resamplers.as_ref().map(|r| r.frames_max()).unwrap_or(0);
        let max_frames = max_frames.max(system_max_num_frames).max(user_num_frames);

        let (input_prod, mut input_cons) =
            create_shared_ringbuffer(num_channels as usize, max_frames * 20, 0);
        let (mut output_prod, output_cons) =
            create_shared_ringbuffer(num_channels as usize, max_frames * 20, 0);
        self.input_prod = input_prod;
        self.output_cons = output_cons;

        self.shutdown_tread();

        let terminate_flag = Arc::clone(&self.terminate_flag);
        let event = self.event.clone();

        self.handle = Some(std::thread::spawn(move || loop {
            if let Some(resamplers) = resamplers.as_mut() {
                // with resampling
                while input_cons.num_frames_stored() >= resamplers.input_frames_next() {
                    if terminate_flag.load(Ordering::SeqCst) {
                        break;
                    }
                    assert!(input_cons.pop_block(resamplers.input_block()));
                    resamplers.process_input(process_block.view_mut());

                    process_fn(&mut process_block);

                    resamplers.process_output(process_block.view());
                    assert!(output_prod.push_block(resamplers.output_block()));
                }
            } else {
                // without resampling
                while input_cons.num_frames_stored() >= user_num_frames {
                    if terminate_flag.load(Ordering::SeqCst) {
                        break;
                    }
                    assert!(input_cons.pop_block(process_block.view_mut()));

                    process_fn(&mut process_block);

                    assert!(output_prod.push_block(process_block.view()));
                }
            }

            // if not enough data is there to do the processing, wait for new data to arrive in audio thread
            let listener = event.listen();
            listener.wait();

            // make this thrad end, if the program is shutting down
            // make sure that the listener gets notified after setting the terminate flag to true!
            if terminate_flag.load(Ordering::SeqCst) {
                break;
            }
        }));
    }

    #[nonblocking]
    pub fn process(&mut self, mut block: impl AudioBlockMut<F>) {
        self.input_prod.push_block(block.view());
        // notify background thread that new data is available
        self.event.notify(usize::MAX);

        if self.output_cons.num_frames_stored() >= block.num_frames() {
            assert!(self.output_cons.pop_block(block.view_mut()));
        } else {
            // what to do if no buffer is there?
            block.clear();
        }
    }

    fn shutdown_tread(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.terminate_flag.store(true, Ordering::SeqCst);
            self.event.notify(usize::MAX);
            handle.join().unwrap();
            self.terminate_flag.store(false, Ordering::SeqCst);
        }
    }
}

impl<F: Float> Drop for AsyncAdapter<F> {
    fn drop(&mut self) {
        self.shutdown_tread();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_async_adapter() {
        let called_flag = Arc::new(AtomicBool::new(false));
        let called_flag_clone = Arc::clone(&called_flag);

        let num_channels = 2;
        let user_num_frames = 512;
        let user_sample_rate = 48000;
        let system_num_frames = 512;
        let system_sample_rate = 48000;
        {
            let mut adapter = AsyncAdapter::<f32>::default();

            let mut counter = 0;

            adapter.prepare(
                num_channels,
                system_sample_rate,
                system_num_frames,
                user_sample_rate,
                user_num_frames,
                move |block| {
                    assert_eq!(block.num_frames(), user_num_frames);
                    called_flag_clone.store(true, Ordering::Relaxed);
                    counter += 1;
                    println!("{counter} - I've been called!");
                },
            );

            let mut block = Stacked::new(num_channels, system_num_frames);
            for _ in 0..1000 {
                adapter.process(block.view_mut());
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        assert!(called_flag.load(Ordering::Relaxed));
    }
}
