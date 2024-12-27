// The resamplers are copied from rubato by Henrik Enquist and adapted to take Blocks

use neort_blocks::{BlockHeap, BlockView, BlockViewMut};

use crate::Float;

use super::{
    base::FftResampler,
    utils::{validate_buffers, validate_sample_rates},
    Resampler,
};

/// A synchronous resampler that needs a varying number of audio frames for input
/// and returns a fixed number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct ResamplerFixedOut<F: Float> {
    num_channels: usize,
    num_frames_out: usize,
    fft_size_in: usize,
    fft_size_out: usize,
    overlaps: BlockHeap<F>,
    output_buffers: BlockHeap<F>,
    saved_frames: usize,
    frames_needed: usize,
    resampler: FftResampler<F>,
}

impl<F: Float> ResamplerFixedOut<F> {
    /// Create a new FftFixedOut.
    ///
    /// Parameters are:
    /// - `sample_rate_input`: Input sample rate, must be > 0.
    /// - `sample_rate_output`: Output sample rate, must be > 0.
    /// - `num_frames_out`: length of output data in frames.
    /// - `sub_chunks`: desired number of subchunks for processing, actual number may be different.
    /// - `num_channels`: number of channels in input/output.
    #[allow(clippy::result_unit_err)]
    pub fn new(
        sample_rate_input: usize,
        sample_rate_output: usize,
        num_frames_out: usize,
        sub_chunks: usize,
        num_channels: usize,
    ) -> Result<Self, ()> {
        validate_sample_rates(sample_rate_input, sample_rate_output)?;

        let gcd = num::integer::gcd(sample_rate_input, sample_rate_output);
        let min_chunk_out = sample_rate_output / gcd;
        let wanted_subsize = num_frames_out / sub_chunks;
        let fft_chunks = (wanted_subsize as f32 / min_chunk_out as f32).ceil() as usize;
        let fft_size_out = fft_chunks * sample_rate_output / gcd;
        let fft_size_in = fft_chunks * sample_rate_input / gcd;

        let resampler = FftResampler::<F>::new(fft_size_in, fft_size_out);

        let overlaps = BlockHeap::new(num_channels, fft_size_out);
        let output_buffers = BlockHeap::new(num_channels, num_frames_out + fft_size_out);

        let saved_frames = 0;
        let chunks_needed = (num_frames_out as f32 / fft_size_out as f32).ceil() as usize;
        let frames_needed = chunks_needed * fft_size_in;

        Ok(ResamplerFixedOut {
            num_channels,
            num_frames_out,
            fft_size_in,
            fft_size_out,
            overlaps,
            output_buffers,
            saved_frames,
            frames_needed,
            resampler,
        })
    }
}

impl<F: Float> Resampler<F> for ResamplerFixedOut<F> {
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
            self.frames_needed,
            self.num_frames_out,
        )
        .unwrap();

        debug_assert!(self.num_frames_out <= output.num_frames());

        // for ((input_ch, out_buf_ch), overlap) in input
        //     .channels()
        //     .into_iter()
        //     .zip(self.output_buffers.iter_mut())
        //     .zip(self.overlaps.iter_mut())
        // {
        //     for (in_chunk, out_chunk) in input_ch
        //         .slice(s![..self.frames_needed])
        //         .exact_chunks(self.fft_size_in)
        //         .into_iter()
        //         .zip(out_buf_ch[self.saved_frames..].chunks_mut(self.fft_size_out))
        //     {
        //         self.resampler.resample_unit(
        //             in_chunk,
        //             ArrayViewMut1::from_shape(
        //                 self.fft_size_out,
        //                 &mut out_chunk[..self.fft_size_out],
        //             )
        //             .unwrap(),
        //             overlap,
        //         );
        //     }
        // }

        for ch in 0..input.num_channels() {
            for (in_chunk, out_chunk) in input[ch][..self.frames_needed]
                .chunks_exact(self.fft_size_in)
                .zip(self.output_buffers[ch][self.saved_frames..].chunks_mut(self.fft_size_out))
            {
                self.resampler.resample_unit(
                    in_chunk,
                    &mut out_chunk[..self.fft_size_out],
                    &mut self.overlaps[ch],
                );
            }
        }

        let processed_frames =
            self.saved_frames + self.fft_size_out * (self.frames_needed / self.fft_size_in);

        // Copy to output, and save extra frames for next round.
        if processed_frames >= self.num_frames_out {
            self.saved_frames = processed_frames - self.num_frames_out;
            // for (mut output_ch, out_buf_ch) in output
            //     .channels_mut()
            //     .into_iter()
            //     .zip(self.output_buffers.iter_mut())
            // {
            //     output_ch
            //         .iter_mut()
            //         .take(self.num_frames_out)
            //         .zip(out_buf_ch.iter())
            //         .for_each(|(a, b)| *a = *b);
            //     out_buf_ch.copy_within(
            //         self.num_frames_out..(self.num_frames_out + self.saved_frames),
            //         0,
            //     );
            // }
            for ch in 0..output.num_channels() {
                output[ch][..self.num_frames_out]
                    .copy_from_slice(&self.output_buffers[ch][..self.num_frames_out]);
                self.output_buffers[ch].copy_within(
                    self.num_frames_out..(self.num_frames_out + self.saved_frames),
                    0,
                );
            }
        } else {
            self.saved_frames = processed_frames;
        }
        // Calculate number of needed frames from next round.
        let frames_needed_out = if self.num_frames_out > self.saved_frames {
            self.num_frames_out - self.saved_frames
        } else {
            0
        };
        let input_frames_used = self.frames_needed;
        let chunks_needed = (frames_needed_out as f32 / self.fft_size_out as f32).ceil() as usize;
        self.frames_needed = chunks_needed * self.fft_size_in;
        Ok((input_frames_used, self.num_frames_out))
    }

    fn input_frames_max(&self) -> usize {
        (self.num_frames_out as f32 / self.fft_size_out as f32).ceil() as usize * self.fft_size_in
    }

    fn input_frames_next(&self) -> usize {
        self.frames_needed
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
        self.fft_size_out / 2
    }

    fn reset(&mut self) {
        self.overlaps.clear();
        self.output_buffers.clear();
        self.saved_frames = 0;
        let chunks_needed = (self.num_frames_out as f32 / self.fft_size_out as f32).ceil() as usize;
        self.frames_needed = chunks_needed * self.fft_size_in;
    }
}

#[cfg(test)]
mod tests {
    use rubato::Resampler as _;

    use super::*;

    #[test]
    fn resampler_fo() {
        let mut resampler = ResamplerFixedOut::<f32>::new(44100, 48000, 1024, 1, 2).unwrap();

        let mut input = resampler.generate_input_block();
        input[0][0] = 1.0;
        input[1][2] = 1.0;
        let mut output = resampler.generate_output_block();

        resampler.process(input.view(), output.view_mut()).unwrap();
        resampler.process(input.view(), output.view_mut()).unwrap();

        // dbg!(&output);

        let mut rub = rubato::FftFixedOut::<f32>::new(44100, 48000, 1024, 1, 2).unwrap();
        let mut rub_in = rub.input_buffer_allocate(true);
        rub_in[0][0] = 1.0;
        rub_in[1][2] = 1.0;

        let _ = rub.process(&rub_in, None).unwrap();
        let rub_output = rub.process(&rub_in, None).unwrap();

        assert_eq!(output.channel(0).to_vec(), rub_output[0]);
        assert_eq!(output.channel(1).to_vec(), rub_output[1]);
    }
}
