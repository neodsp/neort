// The resamplers are copied from rubato by Henrik Enquist and adapted to take Blocks

use neort_blocks::{BlockHeap, BlockView, BlockViewMut};
use neort_float::Float;

use super::{
    base::FftResampler,
    utils::{validate_buffers, validate_sample_rates},
    Resampler,
};

/// A synchronous resampler that accepts a fixed number of audio frames for input
/// and returns a fixed number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct ResamplerFixedInOut<F: Float> {
    num_channels: usize,
    num_frames_in: usize,
    num_frames_out: usize,
    fft_size_in: usize,
    overlaps: BlockHeap<F>,
    resampler: FftResampler<F>,
}

impl<F: Float> ResamplerFixedInOut<F> {
    /// Create a new FftFixedInOut.
    ///
    /// Parameters are:
    /// - `sample_rate_input`: Input sample rate, must be > 0.
    /// - `sample_rate_output`: Output sample rate, must be > 0.
    /// - `frame_size_in`: desired length of input data in frames, actual value may be different.
    /// - `num_channels`: number of channels in input/output.
    #[allow(clippy::result_unit_err)]
    pub fn new(
        sample_rate_input: usize,
        sample_rate_output: usize,
        frame_size_in: usize,
        num_channels: usize,
    ) -> Result<Self, ()> {
        validate_sample_rates(sample_rate_input, sample_rate_output)?;

        let gcd = num_integer::gcd(sample_rate_input, sample_rate_output);
        let min_chunk_in = sample_rate_input / gcd;
        let fft_chunks = (frame_size_in as f32 / min_chunk_in as f32).ceil() as usize;
        let fft_size_out = fft_chunks * sample_rate_output / gcd;
        let fft_size_in = fft_chunks * sample_rate_input / gcd;

        let resampler = FftResampler::<F>::new(fft_size_in, fft_size_out);

        let overlaps = BlockHeap::new(num_channels, fft_size_out);

        Ok(ResamplerFixedInOut {
            num_channels,
            num_frames_in: fft_size_in,
            num_frames_out: fft_size_out,
            fft_size_in,
            overlaps,
            resampler,
        })
    }
}

impl<F: Float> Resampler<F> for ResamplerFixedInOut<F> {
    #[rtsan::nonblocking]
    fn process(
        &mut self,
        input: BlockView<F>,
        mut output: BlockViewMut<F>,
    ) -> Result<(usize, usize), ()> {
        validate_buffers(
            input.view(),
            output.view_mut(),
            self.num_channels,
            self.num_frames_in,
            self.num_frames_out,
        )
        .unwrap();

        for ((output, input), overlap) in output
            .channels_mut()
            .zip(input.channels())
            .zip(self.overlaps.channels_mut())
        {
            self.resampler.resample_unit(input, output, overlap);
        }

        Ok((self.num_frames_in, self.num_frames_out))
    }

    fn input_frames_max(&self) -> usize {
        self.fft_size_in
    }

    fn input_frames_next(&self) -> usize {
        self.fft_size_in
    }

    fn num_channels(&self) -> usize {
        self.num_channels
    }

    fn output_frames_max(&self) -> usize {
        self.num_frames_out
    }

    fn output_frames_next(&self) -> usize {
        self.output_frames_max()
    }

    fn output_delay(&self) -> usize {
        self.num_frames_out / 2
    }

    fn reset(&mut self) {
        self.overlaps.clear();
    }
}

#[cfg(test)]
mod tests {
    use rubato::Resampler as _;

    use crate::resampler::Resampler;

    use super::*;

    #[test]
    fn resampler_fio() {
        let mut resampler = ResamplerFixedInOut::<f32>::new(44100, 48000, 1, 2).unwrap();

        let mut input = resampler.generate_input_block();
        input.channel_mut(0)[0] = 1.0;
        input.channel_mut(1)[2] = 1.0;
        let mut output = resampler.generate_output_block();

        resampler.process(input.view(), output.view_mut()).unwrap();

        let mut rub = rubato::FftFixedInOut::<f32>::new(44100, 48000, 1, 2).unwrap();
        let mut rub_in = rub.input_buffer_allocate(true);
        rub_in[0][0] = 1.0;
        rub_in[1][2] = 1.0;

        let rub_output = rub.process(&rub_in, None).unwrap();

        assert_eq!(output.channel(0).to_vec(), rub_output[0]);
        assert_eq!(output.channel(1).to_vec(), rub_output[1]);
    }
}
