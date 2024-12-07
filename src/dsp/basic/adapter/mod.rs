use num::Float;
use realfft::FftNum;

use crate::{
    audio_block::{Block, BlockRead, BlockWrite},
    ringbuffer::Ringbuffer,
};

use super::resampler::{Resampler, ResamplerFixedIn, ResamplerFixedOut};

pub struct Resamplers<F: Float + FftNum> {
    input: ResamplerFixedOut<F>,
    output: ResamplerFixedIn<F>,
    input_block: Block<F>,
    output_block: Block<F>,
}

impl<F: Float + FftNum> Resamplers<F> {
    pub fn new(
        num_channels: u16,
        system_sample_rate: usize,
        user_sample_rate: usize,
        user_num_frames: usize,
    ) -> Self {
        let input = ResamplerFixedOut::new(
            system_sample_rate,
            user_sample_rate,
            user_num_frames,
            1,
            num_channels,
        )
        .unwrap();
        let output = ResamplerFixedIn::new(
            user_sample_rate,
            system_sample_rate,
            user_num_frames,
            1,
            num_channels,
        )
        .unwrap();

        Self {
            input_block: input.generate_input_block(),
            output_block: output.generate_output_block(),
            input,
            output,
        }
    }

    pub fn process_input(&mut self, output: &mut impl BlockWrite<F>) {
        self.input.process(&self.input_block, output).unwrap();
    }

    pub fn process_output(&mut self, input: &impl BlockRead<F>) {
        self.output.process(input, &mut self.output_block).unwrap();
    }

    pub fn input_block(&mut self) -> &mut Block<F> {
        &mut self.input_block
    }

    pub fn output_block(&self) -> &Block<F> {
        &self.output_block
    }

    pub fn frames_max(&self) -> usize {
        self.input
            .input_frames_max()
            .max(self.output.output_frames_max())
    }
}

#[derive(Default)]
pub struct Adapter<F: Float + FftNum> {
    input_rb: Ringbuffer<F>,
    output_rb: Ringbuffer<F>,
    resamplers: Option<Resamplers<F>>,
    process_block: Block<F>,
    num_frames: usize,
}

impl<F: Float + FftNum> Adapter<F> {
    pub fn prepare(
        &mut self,
        num_channels: u16,
        system_sample_rate: usize,
        system_max_num_frames: usize,
        user_sample_rate: usize,
        user_num_frames: usize,
    ) {
        self.num_frames = user_num_frames;

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

        self.input_rb.prepare(num_channels, max_frames * 3, 0);
        self.output_rb.prepare(num_channels, max_frames * 2, 0);
    }

    fn process(
        &mut self,
        block: &mut impl BlockWrite<F>,
        mut process_fn: impl FnMut(&mut Block<F>),
    ) {
        self.input_rb.push_block(block);

        if let Some(resamplers) = self.resamplers.as_mut() {
            while self.input_rb.num_frames_stored() >= resamplers.input.input_frames_next() {
                self.input_rb.pop_block(resamplers.input_block());
                resamplers.process_input(&mut self.process_block);

                process_fn(&mut self.process_block);

                resamplers.process_output(&self.process_block);
                self.output_rb.push_block(resamplers.output_block());
            }
        } else {
            while self.input_rb.num_frames_stored() >= self.num_frames {
                self.input_rb.pop_block(&mut self.process_block);
                process_fn(&mut self.process_block);
                self.output_rb.push_block(&self.process_block);
            }
        }

        if self.output_rb.num_frames_stored() >= block.num_frames() {
            self.output_rb.pop_block(block);
        }
    }
}
