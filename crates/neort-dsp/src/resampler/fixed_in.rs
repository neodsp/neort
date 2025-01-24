// The resamplers are copied from rubato by Henrik Enquist and adapted to take Blocks

use neort_blocks::{BlockHeap, BlockView, BlockViewMut};
use neort_float::Float;
use rtsan_standalone::nonblocking;

use super::{
    base::FftResampler,
    utils::{validate_buffers, validate_sample_rates},
    Resampler,
};

/// A synchronous resampler that needs a fixed number of audio frames for input
/// and returns a variable number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct ResamplerFixedIn<F: Float> {
    num_channels: usize,
    num_frames_in: usize,
    fft_size_in: usize,
    fft_size_out: usize,
    overlaps: BlockHeap<F>,
    input_buffers: BlockHeap<F>,
    saved_frames: usize,
    resampler: FftResampler<F>,
}

impl<F: Float> ResamplerFixedIn<F> {
    /// Create a new FftFixedIn.
    ///
    /// Parameters are:
    /// - `sample_rate_input`: Input sample rate, must be > 0.
    /// - `sample_rate_output`: Output sample rate, must be > 0.
    /// - `num_frames_in`: length of input data in frames.
    /// - `sub_chunks`: desired number of subchunks for processing, actual number used may be different.
    /// - `num_channels`: number of channels in input/output.
    #[allow(clippy::result_unit_err)]
    pub fn new(
        sample_rate_input: usize,
        sample_rate_output: usize,
        num_frames_in: usize,
        sub_chunks: usize,
        num_channels: usize,
    ) -> Result<Self, ()> {
        validate_sample_rates(sample_rate_input, sample_rate_output)?;

        let gcd = num_integer::gcd(sample_rate_input, sample_rate_output);
        let min_chunk_in = sample_rate_input / gcd;
        let wanted_subsize = num_frames_in / sub_chunks;
        let fft_chunks = (wanted_subsize as f32 / min_chunk_in as f32).ceil() as usize;
        let fft_size_out = fft_chunks * sample_rate_output / gcd;
        let fft_size_in = fft_chunks * sample_rate_input / gcd;

        let resampler = FftResampler::<F>::new(fft_size_in, fft_size_out);

        let overlaps = BlockHeap::new(num_channels, fft_size_out);
        let input_buffers = BlockHeap::new(num_channels, num_frames_in + fft_size_in);

        let saved_frames = 0;

        Ok(ResamplerFixedIn {
            num_channels,
            num_frames_in,
            fft_size_in,
            fft_size_out,
            overlaps,
            input_buffers,
            saved_frames,
            resampler,
        })
    }
}

impl<F: Float> Resampler<F> for ResamplerFixedIn<F> {
    #[nonblocking]
    fn process(
        &mut self,
        input: BlockView<F>,
        mut output: BlockViewMut<F>,
    ) -> Result<(usize, usize), ()> {
        let next_saved_frames = self.saved_frames + self.num_frames_in;
        let num_chunks_ready =
            (next_saved_frames as f32 / self.fft_size_in as f32).floor() as usize;
        let needed_len = num_chunks_ready * self.fft_size_out;

        validate_buffers(
            input.view(),
            output.view_mut(),
            self.num_channels,
            self.num_frames_in,
            needed_len,
        )
        .unwrap();

        // Copy new samples to input buffer.
        // for (input_ch, in_buf) in input
        //     .channels()
        //     .into_iter()
        //     .zip(self.input_buffers.iter_mut())
        // {
        //     in_buf
        //         .iter_mut()
        //         .skip(self.saved_frames)
        //         .take(self.num_frames_in)
        //         .zip(input_ch.iter())
        //         .for_each(|(a, b)| *a = *b);
        // }

        for ch in 0..input.num_channels() {
            for frame in 0..self.num_frames_in {
                self.input_buffers[ch][self.saved_frames + frame] = input[ch][frame];
            }
        }

        self.saved_frames = next_saved_frames;

        let out_frames = output.num_frames();
        debug_assert!(needed_len <= out_frames);

        // for ((input_ch, mut output_ch), overlap) in self
        //     .input_buffers
        //     .iter()
        //     .zip(output.channels_mut())
        //     .zip(self.overlaps.iter_mut())
        // {
        //     for (in_chunk, out_chunk) in input_ch
        //         .chunks(self.fft_size_in)
        //         .take(num_chunks_ready)
        //         .zip(output_ch.exact_chunks_mut(self.fft_size_out))
        //     {
        //         self.resampler.resample_unit(
        //             ArrayView1::from_shape(self.fft_size_in, &in_chunk[..self.fft_size_in])
        //                 .unwrap(),
        //             out_chunk,
        //             overlap,
        //         );
        //     }
        // }

        for ch in 0..self.input_buffers.num_channels() {
            for (in_chunk, out_chunk) in self.input_buffers[ch]
                .chunks(self.fft_size_in)
                .take(num_chunks_ready)
                .zip(output[ch].chunks_exact_mut(self.fft_size_out))
            {
                self.resampler.resample_unit(
                    &in_chunk[..self.fft_size_in],
                    out_chunk,
                    &mut self.overlaps[ch],
                );
            }
        }

        // Save extra frames for next round.
        let frames_in_used = num_chunks_ready * self.fft_size_in;
        let extra = self.saved_frames - frames_in_used;

        if self.saved_frames > frames_in_used {
            for ib_ch in self.input_buffers.channels_mut() {
                ib_ch.copy_within(frames_in_used..self.saved_frames, 0);
            }
        }
        self.saved_frames = extra;
        Ok((self.num_frames_in, needed_len))
    }

    fn input_frames_max(&self) -> usize {
        self.num_frames_in
    }

    fn input_frames_next(&self) -> usize {
        self.num_frames_in
    }

    fn num_channels(&self) -> usize {
        self.num_channels
    }

    fn output_frames_max(&self) -> usize {
        let max_stored_frames = self.fft_size_in - 1;
        let max_available_frames = max_stored_frames + self.num_frames_in;
        let max_subchunks_to_process = max_available_frames / self.fft_size_in;
        max_subchunks_to_process * self.fft_size_out
    }

    fn output_frames_next(&self) -> usize {
        (((self.saved_frames + self.num_frames_in) as f32) / self.fft_size_in as f32).floor()
            as usize
            * self.fft_size_out
    }

    fn output_delay(&self) -> usize {
        self.fft_size_out / 2
    }

    fn reset(&mut self) {
        self.overlaps.clear();
        self.input_buffers.clear();
        self.saved_frames = 0;
    }
}

#[cfg(test)]
mod tests {
    use rubato::Resampler as _;

    use super::*;

    #[test]
    fn resampler_fi() {
        let mut resampler = ResamplerFixedIn::<f32>::new(44100, 48000, 1024, 1, 2).unwrap();

        let mut input = resampler.generate_input_block();
        input[0][0] = 1.0;
        input[1][2] = 1.0;
        let mut output = resampler.generate_output_block();

        resampler.process(input.view(), output.view_mut()).unwrap();
        resampler.process(input.view(), output.view_mut()).unwrap();

        let mut rub = rubato::FftFixedIn::<f32>::new(44100, 48000, 1024, 1, 2).unwrap();
        let mut rub_in = rub.input_buffer_allocate(true);
        rub_in[0][0] = 1.0;
        rub_in[1][2] = 1.0;

        let _ = rub.process(&rub_in, None).unwrap();
        let rub_output = rub.process(&rub_in, None).unwrap();

        assert_eq!(output.channel(0).to_vec(), rub_output[0]);
        assert_eq!(output.channel(1).to_vec(), rub_output[1]);
    }
}
