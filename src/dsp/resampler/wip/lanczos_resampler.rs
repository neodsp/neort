// copied from babycat

use ndarray::ArrayView1;
use num::Float;

use crate::{
    audio_block::{Block, BlockRead, BlockWrite},
    ringbuffer::Ringbuffer,
};

const KERNEL_A: i32 = 5;

pub fn generate_output_block<F: Float>(
    input_sample_rate: f64,
    input_num_frames: u32,
    output_sample_rate: f64,
) -> Block<F> {
    Block::new(
        2,
        calculate_output_frames(input_sample_rate, input_num_frames, output_sample_rate),
    )
}

pub fn calculate_output_frames(
    input_sample_rate: f64,
    input_num_frames: u32,
    output_sample_rate: f64,
) -> u32 {
    (input_num_frames as f64 * output_sample_rate / input_sample_rate).ceil() as u32
}

fn lanczos_kernel<F: Float>(x: F, a: F) -> F {
    if x.is_zero() {
        return F::one();
    }
    let pi = F::from(std::f64::consts::PI).unwrap();
    if -a <= x && x < a {
        return (a * (pi * x).sin() * (pi * x / a).sin()) / (pi * pi * x * x);
    }
    F::zero()
}

pub fn compute_sample<F: Float>(input: ArrayView1<F>, frame_idx: F) -> F {
    let num_input_frames = input.len();
    let x_floor = frame_idx.to_i64().unwrap();
    let i_start = x_floor - KERNEL_A as i64 + 1;
    let i_end = x_floor + KERNEL_A as i64 + 1;
    let mut output = F::zero();
    for i in i_start..i_end {
        if (i as usize) < num_input_frames {
            output = output
                + input[i as usize]
                    * F::from(lanczos_kernel(
                        frame_idx - F::from(i).unwrap(),
                        F::from(KERNEL_A).unwrap(),
                    ))
                    .unwrap();
        }
    }
    output
}

// #[rtsan::nonblocking]
// pub fn process<F: Float>(
//     input: &impl BlockRead<F>,
//     input_sample_rate: f64,
//     num_input_frames: u32,
//     output: &mut impl BlockWrite<F>,
//     output_sample_rate: f64,
// ) -> u32 {
//     let output_sample_rate = output_sample_rate;
//     let num_output_frames =
//         calculate_output_frames(input_sample_rate, num_input_frames, output_sample_rate);
//     assert_eq!(output.num_frames(), num_output_frames);
//     for (mut output_ch, input_ch) in output.channels_mut().into_iter().zip(input.channels()) {
//         for in_frame_id in 0..num_output_frames {
//             let frame_idx =
//                 F::from(in_frame_id as f64 * input_sample_rate / output_sample_rate).unwrap();
//             output_ch[in_frame_id as usize] = compute_sample(input_ch, frame_idx);
//         }
//     }
//     num_output_frames
// }

#[derive(Default)]
pub struct Resampler<F: Float> {
    ringbuffer: Ringbuffer<F>,
    num_channels: u16,
    input_sample_rate: f64,
    output_sample_rate: f64,
}

impl<F: Float> Resampler<F> {
    pub fn prepare(
        &mut self,
        num_channels: u16,
        input_sample_rate: f64,
        input_max_num_frames: u32,
        output_sample_rate: f64,
        output_max_num_frames: u32,
    ) {
        self.input_sample_rate = input_sample_rate;
        self.output_sample_rate = output_sample_rate;
        self.num_channels = num_channels;

        let output_max_num_frames_rs = calculate_output_frames(
            self.input_sample_rate,
            input_max_num_frames,
            self.output_sample_rate,
        );

        let rb_capacity = output_max_num_frames_rs.max(output_max_num_frames) as usize * 100;

        self.ringbuffer.prepare(self.num_channels, rb_capacity, 0);
    }

    #[rtsan::nonblocking]
    pub fn push_block(&mut self, input: &impl BlockRead<F>) {
        let num_output_frames = calculate_output_frames(
            self.input_sample_rate,
            input.num_frames(),
            self.output_sample_rate,
        );
        for ch in 0..self.num_channels {
            for in_frame_id in 0..num_output_frames {
                let frame_idx =
                    F::from(in_frame_id as f64 * self.input_sample_rate / self.output_sample_rate)
                        .unwrap();
                let sample = compute_sample(input.channel(ch), frame_idx);
                assert!(self.ringbuffer.push_sample(ch, sample));
            }
        }
    }

    #[rtsan::nonblocking]
    pub fn pull_block(&mut self, block: &mut impl BlockWrite<F>) {
        if self.ringbuffer.num_stored() >= block.num_frames() as usize {
            self.ringbuffer.pop_block(block, block.num_frames());
        } else {
            block.clear();
        }
    }

    pub fn num_stored(&self) -> usize {
        self.ringbuffer.num_stored()
    }
}

#[cfg(test)]
mod tests {
    use babycat::{
        constants::{RESAMPLE_MODE_BABYCAT_LANCZOS, RESAMPLE_MODE_BABYCAT_SINC},
        Waveform, WaveformArgs,
    };

    use crate::audio_block::{Block, BlockView};

    use super::*;

    #[test]
    fn lanczos_resampler_processor() {
        let mut resampler = Resampler::default();

        resampler.prepare(2, 41000.0, 10, 48000.0, 11);

        let mut input_block = Block::<f32>::new(2, 10);
        let mut output_block = Block::<f32>::new(2, 11);

        *input_block.sample_mut(0, 0) = 1.0;
        *input_block.sample_mut(1, 2) = 1.0;

        resampler.push_block(&input_block);
        resampler.pull_block(&mut output_block);

        let mut data = vec![0.0; 20];
        data[0] = 1.0;
        data[5] = 1.0;
        let a = babycat::Waveform::new(44100, 2, data);

        let expected_output = a
            .resample_by_mode(48000, RESAMPLE_MODE_BABYCAT_LANCZOS)
            .unwrap();

        let expected_block = BlockView::from_buffer(
            expected_output.to_interleaved_samples(),
            2,
            11,
            crate::audio_block::BufferLayout::Interleaved,
        );

        assert_eq!(output_block.view(), expected_block);
    }

    #[test]
    fn long_test() {
        // let mut resampler = Resampler::default();

        // resampler.prepare(2, 41000.0, 10, 48000.0, 11);

        // let input_block = Block::<f32>::new(2, 10);

        // resampler.push_block(&input_block);

        let file = Waveform::from_file("data/sweep.wav", WaveformArgs::default()).unwrap();
        let file = file
            .resample_by_mode(48000, RESAMPLE_MODE_BABYCAT_SINC)
            .unwrap();

        file.to_wav_file("data/sweep_sinc.wav").unwrap();
    }
}
