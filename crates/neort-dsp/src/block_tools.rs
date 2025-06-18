use audio_blocks::Ops;
use audio_blocks::{AudioBlock, AudioBlockMut};
use neort_float::{Float, IntoGeneric};
use rtsan_standalone::nonblocking;

use crate::utils::db_to_gain;

#[nonblocking]
pub fn mix_down_to_mono<F: Float>(
    input: impl AudioBlock<F>,
    mut mono_output: impl AudioBlockMut<F>,
) {
    assert_eq!(mono_output.num_channels(), 1);
    assert_eq!(input.num_frames(), mono_output.num_frames());
    mono_output.channel_mut(0).for_each(|s| *s = F::zero());
    for ch in input.channels() {
        for (mono_sample, in_sample) in mono_output.channel_mut(0).zip(ch) {
            *mono_sample += *in_sample;
        }
    }
    let normalize_factor = F::one() / input.num_channels().as_f();
    for mono_sample in mono_output.channel_mut(0) {
        *mono_sample *= normalize_factor;
    }
}

#[nonblocking]
pub fn duplicate_mono_to_channels<F: Float>(
    mono_input: impl AudioBlock<F>,
    mut output: impl AudioBlockMut<F>,
) {
    assert_eq!(mono_input.num_channels(), 1);
    assert_eq!(mono_input.num_frames(), output.num_frames());
    for channel in output.channels_mut() {
        channel
            .zip(mono_input.channel(0))
            .for_each(|(a, b)| *a = *b);
    }
}

#[nonblocking]
pub fn apply_gain_db<F: Float>(mut block: impl AudioBlockMut<F>, gain_db: f32) {
    let gain = db_to_gain(gain_db).as_f();
    block.for_each(|s| *s *= gain);
}

#[cfg(test)]
mod tests {
    use audio_blocks::{Stacked, StackedView, StackedViewMut};

    use super::*;

    #[test]
    fn test_mix_down_to_mono() {
        let input = StackedView::from_slice(&[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
        let mut output = Stacked::<f32>::new(1, 4);

        mix_down_to_mono(input.view(), output.view_mut());

        // Expected: average of each pair of samples
        assert_eq!(output.channel_slice(0).unwrap(), &[3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_duplicate_mono_to_channels() {
        let input = StackedView::from_slice(&[[1.0, 2.0, 3.0]]);
        let mut output = Stacked::<f32>::new(2, 3);

        duplicate_mono_to_channels(input.view(), output.view_mut());

        assert_eq!(output.channel_slice(0).unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(output.channel_slice(1).unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_apply_gain_db() {
        let mut data = [[1.0, 2.0, 3.0]];
        let mut block = StackedViewMut::from_slice(&mut data);

        apply_gain_db(block.view_mut(), 6.0); // +6dB should approximately double amplitude

        let expected: Vec<f32> = vec![1.0, 2.0, 3.0].iter().map(|x| x * 2.0).collect();
        for (actual, expected) in block.channel(0).zip(expected.iter()) {
            assert!((actual - expected).abs() < 0.02);
        }
    }
}
