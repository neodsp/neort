use crate::{
    audio_block::{Block, BlockRead, BlockView, BlockViewMut, BlockWrite},
    dsp::resampler::{Resampler, ResamplerFixedIn, ResamplerFixedOut},
    Sample,
};

pub struct Resamplers<S: Sample> {
    input: ResamplerFixedOut<S>,
    output: ResamplerFixedIn<S>,
    input_block: Block<S>,
    output_block: Block<S>,
}

impl<S: Sample> Resamplers<S> {
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
        input_block.set_num_frames(input.input_frames_next());
        output_block.set_num_frames(output.output_frames_next());

        Self {
            input_block,
            output_block,
            input,
            output,
        }
    }

    pub fn process_input(&mut self, output: &mut impl BlockWrite<S>) {
        self.input.process(&self.input_block, output).unwrap();
        self.input_block.set_num_frames(self.input_frames_next());
    }

    pub fn process_output(&mut self, input: &impl BlockRead<S>) {
        self.output_block.set_num_frames(self.output_frames_next());
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

    pub fn input_block(&mut self) -> BlockViewMut<S> {
        self.input_block.view_mut()
    }

    pub fn output_block(&self) -> BlockView<S> {
        self.output_block.view()
    }

    pub fn frames_max(&self) -> usize {
        self.input
            .input_frames_max()
            .max(self.output.output_frames_max())
    }
}
