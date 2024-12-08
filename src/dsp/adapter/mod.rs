use num::Float;
use realfft::FftNum;
use resamplers::Resamplers;
use tools::{find_max_index, impulse_response};

use crate::{
    audio_block::{Block, BlockWrite},
    ringbuffer::Ringbuffer,
};

mod resamplers;
mod tools;

#[derive(Default)]
pub struct Adapter<F: Float + FftNum> {
    input_rb: Ringbuffer<F>,
    output_rb: Ringbuffer<F>,
    resamplers: Option<Resamplers<F>>,
    process_block: Block<F>,
    user_num_frames: usize,
}

impl<F: Float + FftNum> Adapter<F> {
    /// Returns the delay the adaptor is expected to have
    pub fn prepare(
        &mut self,
        num_channels: u16,
        system_sample_rate: usize,
        system_max_num_frames: usize,
        user_sample_rate: usize,
        user_num_frames: usize,
    ) -> usize {
        self.user_num_frames = user_num_frames;

        if system_sample_rate != user_sample_rate {
            self.resamplers = Some(Resamplers::new(
                num_channels,
                system_sample_rate,
                user_sample_rate,
                user_num_frames,
            ));
        }

        self.process_block = Block::new(num_channels, user_num_frames);

        let max_frames = self
            .resamplers
            .as_ref()
            .map(|r| r.frames_max())
            .unwrap_or(0);

        let max_frames = max_frames.max(system_max_num_frames).max(user_num_frames);

        self.input_rb.prepare(num_channels, max_frames * 10, 0);
        self.output_rb.prepare(num_channels, max_frames * 10, 0);

        let ir = impulse_response(10, system_max_num_frames, |block| {
            self.process(block, |_| {});
        });
        self.reset();

        let delay = find_max_index(&ir);
        delay
    }

    pub fn process(
        &mut self,
        block: &mut impl BlockWrite<F>,
        mut process_fn: impl FnMut(&mut Block<F>),
    ) {
        assert!(self.input_rb.push_block(block));

        if let Some(resamplers) = self.resamplers.as_mut() {
            // Resampling necessary
            while self.input_rb.num_frames_stored() >= resamplers.input_frames_next() {
                assert!(self.input_rb.pop_block(&mut resamplers.input_block()));
                resamplers.process_input(&mut self.process_block);

                process_fn(&mut self.process_block);

                resamplers.process_output(&self.process_block);
                assert!(self.output_rb.push_block(&resamplers.output_block()));
            }
        } else {
            // Resampling unnecessary
            while self.input_rb.num_frames_stored() >= self.user_num_frames {
                self.input_rb.pop_block(&mut self.process_block);

                process_fn(&mut self.process_block);

                assert!(self.output_rb.push_block(&self.process_block));
            }
        }

        if self.output_rb.num_frames_stored() >= block.num_frames() {
            assert!(self.output_rb.pop_block(block));
        } else {
            block.clear();
        }
    }

    pub fn reset(&mut self) {
        self.resamplers.as_mut().map(|r| r.reset());
        self.input_rb.reset();
        self.output_rb.reset();
    }
}

#[cfg(test)]
mod tests {
    use crate::audio_block::BlockRead;

    use super::*;

    #[test]
    fn adapter() {
        let mut adapter = Adapter::<f32>::default();

        let delay = adapter.prepare(2, 48000, 512, 44100, 512);

        dbg!(delay);

        let mut block = Block::new(2, 512);

        adapter.process(&mut block, |block| {
            assert_eq!(block.num_frames(), 512);
        });
    }
}
