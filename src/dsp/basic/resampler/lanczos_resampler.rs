// copied from babycat

use ndarray::ArrayView1;
use num::Float;

use crate::audio_block::{Block, BlockRead, BlockWrite};

const KERNEL_A: i32 = 5;

pub fn generate_output_block<F: Float>(
    input_sample_rate: f64,
    input_num_frames: u32,
    output_sample_rate: f64,
) -> Block<F> {
    Block::new(
        48000.0,
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

#[rtsan::nonblocking]
pub fn process<F: Float>(
    input: &impl BlockRead<F>,
    num_input_frames: u32,
    output: &mut impl BlockWrite<F>,
) -> u32 {
    let output_sample_rate = output.sample_rate();
    let num_output_frames =
        calculate_output_frames(input.sample_rate(), input.num_frames(), output_sample_rate);
    assert_eq!(output.num_frames(), num_output_frames);
    for (mut output_ch, input_ch) in output.channels_mut().into_iter().zip(input.channels()) {
        for in_frame_id in 0..num_output_frames {
            let frame_idx =
                F::from(in_frame_id as f64 * input.sample_rate() / output_sample_rate).unwrap();
            output_ch[in_frame_id as usize] = compute_sample(input_ch, frame_idx);
        }
    }
    num_output_frames
}

#[cfg(test)]
mod tests {
    use babycat::constants::RESAMPLE_MODE_BABYCAT_LANCZOS;

    use crate::audio_block::{Block, BlockView};

    use super::*;

    #[test]
    fn lanczos_resampler() {
        let mut input_block = Block::<f32>::new(44100.0, 2, 10);
        let mut output_block =
            generate_output_block(input_block.sample_rate(), input_block.num_frames(), 48000.0);

        *input_block.sample_mut(0, 0) = 1.0;
        *input_block.sample_mut(1, 2) = 1.0;

        process(&input_block, input_block.num_frames(), &mut output_block);

        let mut data = vec![0.0; 20];
        data[0] = 1.0;
        data[5] = 1.0;
        let a = babycat::Waveform::new(44100, 2, data);

        let expected_output = a
            .resample_by_mode(48000, RESAMPLE_MODE_BABYCAT_LANCZOS)
            .unwrap();

        let expected_block = BlockView::from_buffer(
            expected_output.to_interleaved_samples(),
            48000.0,
            2,
            11,
            crate::audio_block::BufferLayout::Interleaved,
        );

        assert_eq!(output_block.view(), expected_block);
    }
}
