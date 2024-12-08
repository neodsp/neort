use num::Float;
use realfft::FftNum;

use crate::{
    audio_block::{Block, BlockRead, BlockWrite},
    dsp::resampler::{Resampler, ResamplerFixedIn, ResamplerFixedOut},
};

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

    pub fn input_frames_next(&self) -> usize {
        self.input.input_frames_next()
    }

    pub fn reset(&mut self) {
        self.input.reset();
        self.output.reset();
        self.input_block.clear();
        self.output_block.clear();
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
