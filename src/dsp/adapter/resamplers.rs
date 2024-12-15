use num::Float;
use realfft::FftNum;

use crate::{
    audio_block::{Block, BlockRead, BlockView, BlockViewMut, BlockWrite},
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

        let mut input_block = input.generate_input_block();
        let mut output_block = output.generate_output_block();
        input_block.set_num_frames_accesible(input.input_frames_next());
        output_block.set_num_frames_accesible(output.output_frames_next());

        Self {
            input_block,
            output_block,
            input,
            output,
        }
    }

    pub fn process_input(&mut self, output: &mut impl BlockWrite<F>) {
        self.input
            .process(&self.input_block.view(), output)
            .unwrap();
        self.input_block
            .set_num_frames_accesible(self.input_frames_next());
    }

    pub fn process_output(&mut self, input: &impl BlockRead<F>) {
        self.output_block
            .set_num_frames_accesible(self.output_frames_next());
        self.output
            .process(input, &mut self.output_block.view_mut())
            .unwrap();
    }

    pub fn input_frames_next(&self) -> usize {
        self.input.input_frames_next()
    }

    pub fn output_frames_next(&self) -> usize {
        self.output.output_frames_next()
    }

    pub fn reset(&mut self) {
        self.input.reset();
        self.output.reset();
        self.input_block.clear();
        self.output_block.clear();
    }

    pub fn input_block(&mut self) -> BlockViewMut<F> {
        self.input_block.view_mut()
    }

    pub fn output_block(&self) -> BlockView<F> {
        self.output_block.view()
    }

    pub fn frames_max(&self) -> usize {
        self.input
            .input_frames_max()
            .max(self.output.output_frames_max())
    }
}
