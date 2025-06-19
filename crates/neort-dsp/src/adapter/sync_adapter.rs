use super::resamplers::Resamplers;
use super::tools::impulse_response;
use audio_blocks::{AudioBlock, AudioBlockMut, Ops, Stacked};
use neort_float::Float;

use crate::adapter::tools::find_max_index_per_channel;
use crate::ringbuffer::local::RingbufferLocal;

pub struct SyncAdapter<F: Float> {
    input_rb: RingbufferLocal<F>,
    output_rb: RingbufferLocal<F>,
    resamplers: Option<Resamplers<F>>,
    process_block: Stacked<F>,
    user_num_frames: usize,
}

impl<F: Float> Default for SyncAdapter<F> {
    fn default() -> Self {
        Self {
            input_rb: Default::default(),
            output_rb: Default::default(),
            resamplers: None,
            process_block: Stacked::new(0, 0),
            user_num_frames: 0,
        }
    }
}

impl<F: Float> SyncAdapter<F> {
    /// Returns the delay the adaptor is expected to have
    pub fn prepare(
        &mut self,
        num_channels: u16,
        system_sample_rate: u32,
        system_max_num_frames: usize,
        user_sample_rate: u32,
        user_num_frames: usize,
    ) -> usize {
        self.user_num_frames = user_num_frames;

        if system_sample_rate != user_sample_rate {
            self.resamplers = Some(Resamplers::new(
                num_channels as usize,
                system_sample_rate as usize,
                user_sample_rate as usize,
                user_num_frames,
            ));
        } else {
            self.resamplers = None;
        }

        self.process_block = Stacked::new(num_channels, user_num_frames);

        let max_frames = self
            .resamplers
            .as_ref()
            .map(|r| r.frames_max())
            .unwrap_or(0);

        let max_frames = max_frames.max(system_max_num_frames).max(user_num_frames);

        self.input_rb.prepare(num_channels, max_frames * 10, 0);
        self.output_rb.prepare(num_channels, max_frames * 10, 0);

        let ir = impulse_response(
            10,
            num_channels,
            system_max_num_frames,
            |block: &mut Stacked<F>| {
                self.process(block.view_mut(), |_| {});
            },
        );

        // return delay
        find_max_index_per_channel(ir)[0]
    }

    pub fn process(
        &mut self,
        mut block: impl AudioBlockMut<F>,
        mut process_fn: impl FnMut(&mut Stacked<F>),
    ) {
        assert!(self.input_rb.push_block(block.view()));

        if let Some(resamplers) = self.resamplers.as_mut() {
            // Resampling necessary
            while self.input_rb.num_frames_stored() >= resamplers.input_frames_next() {
                assert!(self.input_rb.pop_block(resamplers.input_block()));
                resamplers.process_input(self.process_block.view_mut());

                process_fn(&mut self.process_block);

                resamplers.process_output(self.process_block.view());
                assert!(self.output_rb.push_block(resamplers.output_block()));
            }
        } else {
            // Resampling unnecessary
            while self.input_rb.num_frames_stored() >= self.user_num_frames {
                self.input_rb.pop_block(self.process_block.view_mut());

                process_fn(&mut self.process_block);

                assert!(self.output_rb.push_block(self.process_block.view()));
            }
        }

        if self.output_rb.num_frames_stored() >= block.num_frames() {
            assert!(self.output_rb.pop_block(block));
        } else {
            block.clear();
        }
    }

    pub fn reset(&mut self) {
        if let Some(r) = self.resamplers.as_mut() {
            r.reset()
        }
        self.input_rb.reset();
        self.output_rb.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter() {
        let mut adapter = SyncAdapter::<f32>::default();

        let delay = adapter.prepare(2, 44100, 512, 48000, 512);

        dbg!(delay);

        let mut block = Stacked::<f32>::new(2, 512);

        adapter.process(block.view_mut(), |block| {
            assert_eq!(block.num_frames(), 512);
        });
    }
}
