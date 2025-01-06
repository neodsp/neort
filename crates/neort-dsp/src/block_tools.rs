use neort_blocks::{BlockView, BlockViewMut};
use neort_float::{Float, IntoGeneric};

use crate::utils::db_to_gain;

pub fn mix_down_to_mono<F: Float>(input: BlockView<F>, mut mono_output: BlockViewMut<F>) {
    assert_eq!(mono_output.num_channels(), 1);
    assert_eq!(input.num_frames(), mono_output.num_frames());
    let mono_channel = mono_output.channel_mut(0);
    mono_channel.fill(F::zero());
    for ch in input.channels() {
        for (mono_sample, in_sample) in mono_channel.iter_mut().zip(ch.iter()) {
            *mono_sample += *in_sample;
        }
    }
    let normalize_factor = F::one() / input.num_channels().as_f();
    for mono_sample in mono_channel {
        *mono_sample *= normalize_factor;
    }
}

pub fn duplicate_mono_to_channels<F: Float>(mono_input: BlockView<F>, mut output: BlockViewMut<F>) {
    assert_eq!(mono_input.num_channels(), 1);
    assert_eq!(mono_input.num_frames(), output.num_frames());
    let mono_channel = mono_input.channel(0);
    for channel in output.channels_mut() {
        channel.copy_from_slice(mono_channel);
    }
}

pub fn apply_gain_db<F: Float>(mut block: BlockViewMut<F>, gain_db: f32) {
    let gain = db_to_gain(gain_db).as_f();
    for sample in block.raw_data_mut() {
        *sample *= gain;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neort_blocks::BlockHeap;

    #[test]
    fn test_mix_down_to_mono() {
        let mut input = BlockHeap::<f32>::new(2, 4);
        let mut output = BlockHeap::<f32>::new(1, 4);

        // Set input values
        input
            .raw_data_mut()
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        mix_down_to_mono(input.view(), output.view_mut());

        // Expected: average of each pair of samples
        assert_eq!(output.raw_data(), &[3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_duplicate_mono_to_channels() {
        let mut input = BlockHeap::<f32>::new(1, 3);
        let mut output = BlockHeap::<f32>::new(2, 3);

        input.raw_data_mut().copy_from_slice(&[1.0, 2.0, 3.0]);

        duplicate_mono_to_channels(input.view(), output.view_mut());

        assert_eq!(output.raw_data(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_apply_gain_db() {
        let mut block = BlockHeap::<f32>::new(1, 3);
        block.raw_data_mut().copy_from_slice(&[1.0, 2.0, 3.0]);

        apply_gain_db(block.view_mut(), 6.0); // +6dB should approximately double amplitude

        let expected: Vec<f32> = vec![1.0, 2.0, 3.0].iter().map(|x| x * 2.0).collect();
        for (actual, expected) in block.raw_data().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 0.02);
        }
    }
}
