// This fft resampler is copied from rubato by Henrik Enquist

use std::sync::Arc;

use num::{Complex, Float, Zero};
use realfft::{ComplexToReal, FftNum, RealFftPlanner, RealToComplex};

/// A helper for resampling a single chunk of data.
struct FftResampler<T> {
    fft_size_in: usize,
    fft_size_out: usize,
    filter_f: Vec<Complex<T>>,
    fft: Arc<dyn RealToComplex<T>>,
    ifft: Arc<dyn ComplexToReal<T>>,
    scratch_fw: Vec<Complex<T>>,
    scratch_inv: Vec<Complex<T>>,
    input_buf: Vec<T>,
    input_f: Vec<Complex<T>>,
    output_f: Vec<Complex<T>>,
    output_buf: Vec<T>,
}

/// A synchronous resampler that needs a fixed number of audio frames for input
/// and returns a variable number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct FftFixedIn<T> {
    nbr_channels: usize,
    chunk_size_in: usize,
    fft_size_in: usize,
    fft_size_out: usize,
    overlaps: Vec<Vec<T>>,
    input_buffers: Vec<Vec<T>>,
    channel_mask: Vec<bool>,
    saved_frames: usize,
    resampler: FftResampler<T>,
}

/// A synchronous resampler that needs a varying number of audio frames for input
/// and returns a fixed number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct FftFixedOut<T> {
    nbr_channels: usize,
    chunk_size_out: usize,
    fft_size_in: usize,
    fft_size_out: usize,
    overlaps: Vec<Vec<T>>,
    output_buffers: Vec<Vec<T>>,
    channel_mask: Vec<bool>,
    saved_frames: usize,
    frames_needed: usize,
    resampler: FftResampler<T>,
}

/// A synchronous resampler that accepts a fixed number of audio frames for input
/// and returns a fixed number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct FftFixedInOut<T> {
    nbr_channels: usize,
    chunk_size_in: usize,
    chunk_size_out: usize,
    fft_size_in: usize,
    channel_mask: Vec<bool>,
    overlaps: Vec<Vec<T>>,
    resampler: FftResampler<T>,
}

fn validate_sample_rates(input: usize, output: usize) -> Result<(), ()> {
    if input == 0 || output == 0 {
        return Err(());
    }
    Ok(())
}

impl<F: Float + FftNum> FftResampler<F> {
    //
    pub fn new(fft_size_in: usize, fft_size_out: usize) -> Self {
        // calculate antialiasing cutoff
        let cutoff = if fft_size_in > fft_size_out {
            calculate_cutoff::<f32>(fft_size_out, WindowFunction::BlackmanHarris2)
                * fft_size_out as f32
                / fft_size_in as f32
        } else {
            calculate_cutoff::<f32>(fft_size_in, WindowFunction::BlackmanHarris2)
        };
        // debug!(
        //     "Create new FftResampler, fft_size_in: {}, fft_size_out: {}, cutoff: {}",
        //     fft_size_in, fft_size_out, cutoff
        // );
        let sinc = make_sincs::<F>(fft_size_in, 1, cutoff, WindowFunction::BlackmanHarris2);
        let mut filter_t: Vec<F> = vec![F::zero(); 2 * fft_size_in];
        let mut filter_f: Vec<Complex<F>> = vec![Complex::zero(); fft_size_in + 1];
        for (n, f) in filter_t.iter_mut().enumerate().take(fft_size_in) {
            *f = sinc[0][n] / F::from(2 * fft_size_in).unwrap();
        }

        let input_f: Vec<Complex<F>> = vec![Complex::zero(); fft_size_in + 1];
        let input_buf: Vec<F> = vec![F::zero(); 2 * fft_size_in];
        let output_f: Vec<Complex<F>> = vec![Complex::zero(); fft_size_out + 1];
        let output_buf: Vec<F> = vec![F::zero(); 2 * fft_size_out];
        let mut planner = RealFftPlanner::<F>::new();
        let fft = planner.plan_fft_forward(2 * fft_size_in);
        let ifft = planner.plan_fft_inverse(2 * fft_size_out);
        fft.process(&mut filter_t, &mut filter_f).unwrap();
        let scratch_fw = fft.make_scratch_vec();
        let scratch_inv = ifft.make_scratch_vec();

        FftResampler {
            fft_size_in,
            fft_size_out,
            filter_f,
            fft,
            ifft,
            scratch_fw,
            scratch_inv,
            input_buf,
            input_f,
            output_f,
            output_buf,
        }
    }

    /// Resample a small chunk.
    fn resample_unit(&mut self, wave_in: &[F], wave_out: &mut [F], overlap: &mut [F]) {
        // Copy to input buffer and clear padding area.
        self.input_buf[0..self.fft_size_in].copy_from_slice(wave_in);
        for item in self
            .input_buf
            .iter_mut()
            .skip(self.fft_size_in)
            .take(self.fft_size_in)
        {
            *item = F::zero();
        }

        // FFT and store result in history, update index.
        self.fft
            .process_with_scratch(&mut self.input_buf, &mut self.input_f, &mut self.scratch_fw)
            .unwrap();

        let new_len = if self.fft_size_in < self.fft_size_out {
            self.fft_size_in + 1
        } else {
            self.fft_size_out
        };

        // Multiply with filter FT.
        self.input_f
            .iter_mut()
            .take(new_len)
            .zip(self.filter_f.iter())
            .for_each(|(spec, filt)| *spec = *spec * filt);

        // copy to modified spectrum
        self.output_f[0..new_len].copy_from_slice(&self.input_f[0..new_len]);
        for val in self.output_f[new_len..].iter_mut() {
            *val = Complex::zero();
        }
        // IFFT result, store result and overlap.
        self.ifft
            .process_with_scratch(
                &mut self.output_f,
                &mut self.output_buf,
                &mut self.scratch_inv,
            )
            .unwrap();
        for (n, item) in wave_out.iter_mut().enumerate().take(self.fft_size_out) {
            *item = self.output_buf[n] + overlap[n];
        }
        overlap.copy_from_slice(&self.output_buf[self.fft_size_out..]);
    }
}

impl<T> FftFixedInOut<T>
where
    T: Float + FftNum,
{
    /// Create a new FftFixedInOut.
    ///
    /// Parameters are:
    /// - `sample_rate_input`: Input sample rate, must be > 0.
    /// - `sample_rate_output`: Output sample rate, must be > 0.
    /// - `chunk_size_in`: desired length of input data in frames, actual value may be different.
    /// - `nbr_channels`: number of channels in input/output.
    pub fn new(
        sample_rate_input: usize,
        sample_rate_output: usize,
        chunk_size_in: usize,
        nbr_channels: usize,
    ) -> Result<Self, ()> {
        validate_sample_rates(sample_rate_input, sample_rate_output)?;

        // debug!(
        //     "Create new FftFixedInOut, sample_rate_input: {}, sample_rate_output: {} chunk_size_in: {}, channels: {}",
        //     sample_rate_input, sample_rate_output, chunk_size_in, nbr_channels
        // );

        let gcd = num::integer::gcd(sample_rate_input, sample_rate_output);
        let min_chunk_in = sample_rate_input / gcd;
        let fft_chunks = (chunk_size_in as f32 / min_chunk_in as f32).ceil() as usize;
        let fft_size_out = fft_chunks * sample_rate_output / gcd;
        let fft_size_in = fft_chunks * sample_rate_input / gcd;

        let resampler = FftResampler::<T>::new(fft_size_in, fft_size_out);

        let overlaps: Vec<Vec<T>> = vec![vec![T::zero(); fft_size_out]; nbr_channels];

        let channel_mask = vec![true; nbr_channels];

        Ok(FftFixedInOut {
            nbr_channels,
            chunk_size_in: fft_size_in,
            chunk_size_out: fft_size_out,
            fft_size_in,
            overlaps,
            resampler,
            channel_mask,
        })
    }
}

impl<T> FftFixedInOut<T>
where
    T: Float + FftNum,
{
    fn process_into_buffer<Vin: AsRef<[T]>, Vout: AsMut<[T]>>(
        &mut self,
        wave_in: &[Vin],
        wave_out: &mut [Vout],
        active_channels_mask: Option<&[bool]>,
    ) -> Result<(usize, usize), ()> {
        if let Some(mask) = active_channels_mask {
            self.channel_mask.copy_from_slice(mask);
        } else {
            update_mask_from_buffers(&mut self.channel_mask);
        };

        validate_buffers(
            wave_in,
            wave_out,
            &self.channel_mask,
            self.nbr_channels,
            self.chunk_size_in,
            self.chunk_size_out,
        )
        .unwrap();

        for (channel, active) in self.channel_mask.iter().enumerate() {
            if *active {
                self.resampler.resample_unit(
                    &wave_in[channel].as_ref()[..self.chunk_size_in],
                    &mut wave_out[channel].as_mut()[..self.chunk_size_out],
                    &mut self.overlaps[channel],
                )
            }
        }
        Ok((self.chunk_size_in, self.chunk_size_out))
    }

    fn input_frames_max(&self) -> usize {
        self.fft_size_in
    }

    fn input_frames_next(&self) -> usize {
        self.fft_size_in
    }

    fn nbr_channels(&self) -> usize {
        self.nbr_channels
    }

    fn output_frames_max(&self) -> usize {
        self.chunk_size_out
    }

    fn output_frames_next(&self) -> usize {
        self.output_frames_max()
    }

    fn output_delay(&self) -> usize {
        self.chunk_size_out / 2
    }

    fn reset(&mut self) {
        self.overlaps
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = T::zero()));
        self.channel_mask.iter_mut().for_each(|val| *val = true);
    }
}

impl<T> FftFixedOut<T>
where
    T: Float + FftNum,
{
    /// Create a new FftFixedOut.
    ///
    /// Parameters are:
    /// - `sample_rate_input`: Input sample rate, must be > 0.
    /// - `sample_rate_output`: Output sample rate, must be > 0.
    /// - `chunk_size_out`: length of output data in frames.
    /// - `sub_chunks`: desired number of subchunks for processing, actual number may be different.
    /// - `nbr_channels`: number of channels in input/output.
    pub fn new(
        sample_rate_input: usize,
        sample_rate_output: usize,
        chunk_size_out: usize,
        sub_chunks: usize,
        nbr_channels: usize,
    ) -> Result<Self, ()> {
        validate_sample_rates(sample_rate_input, sample_rate_output)?;

        let gcd = num::integer::gcd(sample_rate_input, sample_rate_output);
        let min_chunk_out = sample_rate_output / gcd;
        let wanted_subsize = chunk_size_out / sub_chunks;
        let fft_chunks = (wanted_subsize as f32 / min_chunk_out as f32).ceil() as usize;
        let fft_size_out = fft_chunks * sample_rate_output / gcd;
        let fft_size_in = fft_chunks * sample_rate_input / gcd;

        let resampler = FftResampler::<T>::new(fft_size_in, fft_size_out);

        // debug!(
        //     "Create new FftFixedOut, sample_rate_input: {}, sample_rate_output: {} chunk_size_in: {}, channels: {}, fft_size_in: {}, fft_size_out: {}",
        //     sample_rate_input, sample_rate_output, chunk_size_out, nbr_channels, fft_size_in, fft_size_out
        // );

        let overlaps: Vec<Vec<T>> = vec![vec![T::zero(); fft_size_out]; nbr_channels];
        let output_buffers: Vec<Vec<T>> =
            vec![vec![T::zero(); chunk_size_out + fft_size_out]; nbr_channels];

        let channel_mask = vec![true; nbr_channels];

        let saved_frames = 0;
        let chunks_needed = (chunk_size_out as f32 / fft_size_out as f32).ceil() as usize;
        let frames_needed = chunks_needed * fft_size_in;

        Ok(FftFixedOut {
            nbr_channels,
            chunk_size_out,
            fft_size_in,
            fft_size_out,
            overlaps,
            output_buffers,
            saved_frames,
            frames_needed,
            resampler,
            channel_mask,
        })
    }
}

impl<T> FftFixedOut<T>
where
    T: Float + FftNum,
{
    fn process_into_buffer<Vin: AsRef<[T]>, Vout: AsMut<[T]>>(
        &mut self,
        wave_in: &[Vin],
        wave_out: &mut [Vout],
        active_channels_mask: Option<&[bool]>,
    ) -> Result<(usize, usize), ()> {
        if let Some(mask) = active_channels_mask {
            self.channel_mask.copy_from_slice(mask);
        } else {
            update_mask_from_buffers(&mut self.channel_mask);
        };

        validate_buffers(
            wave_in,
            wave_out,
            &self.channel_mask,
            self.nbr_channels,
            self.frames_needed,
            self.chunk_size_out,
        )
        .unwrap();

        for (chan, active) in self.channel_mask.iter().enumerate() {
            if *active {
                debug_assert!(self.chunk_size_out <= wave_out[chan].as_mut().len());
                for (in_chunk, out_chunk) in wave_in[chan].as_ref()[..self.frames_needed]
                    .chunks(self.fft_size_in)
                    .zip(
                        self.output_buffers[chan][self.saved_frames..]
                            .chunks_mut(self.fft_size_out),
                    )
                {
                    self.resampler
                        .resample_unit(in_chunk, out_chunk, &mut self.overlaps[chan]);
                }
            }
        }
        let processed_frames =
            self.saved_frames + self.fft_size_out * (self.frames_needed / self.fft_size_in);

        // Copy to output, and save extra frames for next round.
        if processed_frames >= self.chunk_size_out {
            self.saved_frames = processed_frames - self.chunk_size_out;
            for (chan, active) in self.channel_mask.iter().enumerate() {
                if *active {
                    wave_out[chan].as_mut()[..self.chunk_size_out]
                        .copy_from_slice(&self.output_buffers[chan][..self.chunk_size_out]);
                    self.output_buffers[chan].copy_within(
                        self.chunk_size_out..(self.chunk_size_out + self.saved_frames),
                        0,
                    );
                }
            }
        } else {
            self.saved_frames = processed_frames;
        }
        // Calculate number of needed frames from next round.
        let frames_needed_out = if self.chunk_size_out > self.saved_frames {
            self.chunk_size_out - self.saved_frames
        } else {
            0
        };
        let input_frames_used = self.frames_needed;
        let chunks_needed = (frames_needed_out as f32 / self.fft_size_out as f32).ceil() as usize;
        self.frames_needed = chunks_needed * self.fft_size_in;
        Ok((input_frames_used, self.chunk_size_out))
    }

    fn input_frames_max(&self) -> usize {
        (self.chunk_size_out as f32 / self.fft_size_out as f32).ceil() as usize * self.fft_size_in
    }

    fn input_frames_next(&self) -> usize {
        self.frames_needed
    }

    fn nbr_channels(&self) -> usize {
        self.nbr_channels
    }

    fn output_frames_max(&self) -> usize {
        self.chunk_size_out
    }

    fn output_frames_next(&self) -> usize {
        self.output_frames_max()
    }

    fn output_delay(&self) -> usize {
        self.fft_size_out / 2
    }

    fn reset(&mut self) {
        self.overlaps
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = T::zero()));
        self.output_buffers
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = T::zero()));
        self.channel_mask.iter_mut().for_each(|val| *val = true);
        self.saved_frames = 0;
        let chunks_needed = (self.chunk_size_out as f32 / self.fft_size_out as f32).ceil() as usize;
        self.frames_needed = chunks_needed * self.fft_size_in;
    }
}

impl<T> FftFixedIn<T>
where
    T: Float + FftNum,
{
    /// Create a new FftFixedIn.
    ///
    /// Parameters are:
    /// - `sample_rate_input`: Input sample rate, must be > 0.
    /// - `sample_rate_output`: Output sample rate, must be > 0.
    /// - `chunk_size_in`: length of input data in frames.
    /// - `sub_chunks`: desired number of subchunks for processing, actual number used may be different.
    /// - `nbr_channels`: number of channels in input/output.
    pub fn new(
        sample_rate_input: usize,
        sample_rate_output: usize,
        chunk_size_in: usize,
        sub_chunks: usize,
        nbr_channels: usize,
    ) -> Result<Self, ()> {
        validate_sample_rates(sample_rate_input, sample_rate_output)?;

        let gcd = num::integer::gcd(sample_rate_input, sample_rate_output);
        let min_chunk_in = sample_rate_input / gcd;
        let wanted_subsize = chunk_size_in / sub_chunks;
        let fft_chunks = (wanted_subsize as f32 / min_chunk_in as f32).ceil() as usize;
        let fft_size_out = fft_chunks * sample_rate_output / gcd;
        let fft_size_in = fft_chunks * sample_rate_input / gcd;

        let resampler = FftResampler::<T>::new(fft_size_in, fft_size_out);
        // debug!(
        //     "Create new FftFixedOut, sample_rate_input: {}, sample_rate_output: {} chunk_size_in: {}, channels: {}, fft_size_in: {}, fft_size_out: {}",
        //     sample_rate_input, sample_rate_output, chunk_size_in, nbr_channels, fft_size_in, fft_size_out
        // );

        let overlaps: Vec<Vec<T>> = vec![vec![T::zero(); fft_size_out]; nbr_channels];
        let input_buffers: Vec<Vec<T>> =
            vec![vec![T::zero(); chunk_size_in + fft_size_in]; nbr_channels];

        let channel_mask = vec![true; nbr_channels];

        let saved_frames = 0;

        Ok(FftFixedIn {
            nbr_channels,
            chunk_size_in,
            fft_size_in,
            fft_size_out,
            overlaps,
            input_buffers,
            saved_frames,
            resampler,
            channel_mask,
        })
    }
}

impl<T> FftFixedIn<T>
where
    T: Float + FftNum,
{
    fn process_into_buffer<Vin: AsRef<[T]>, Vout: AsMut<[T]>>(
        &mut self,
        wave_in: &[Vin],
        wave_out: &mut [Vout],
    ) -> Result<(usize, usize), &'static str> {
        let next_saved_frames = self.saved_frames + self.chunk_size_in;
        let nbr_chunks_ready =
            (next_saved_frames as f32 / self.fft_size_in as f32).floor() as usize;
        let needed_len = nbr_chunks_ready * self.fft_size_out;

        validate_buffers(
            wave_in,
            wave_out,
            &self.channel_mask,
            self.nbr_channels,
            self.chunk_size_in,
            needed_len,
        )?;

        // Copy new samples to input buffer.
        for (chan, active) in self.channel_mask.iter().enumerate() {
            if *active {
                for (input, buffer) in wave_in[chan].as_ref().iter().zip(
                    self.input_buffers[chan]
                        .iter_mut()
                        .skip(self.saved_frames)
                        .take(self.chunk_size_in),
                ) {
                    *buffer = *input;
                }
            }
        }

        self.saved_frames = next_saved_frames;

        for (chan, active) in self.channel_mask.iter().enumerate() {
            if *active {
                debug_assert!(needed_len <= wave_out[chan].as_mut().len());
                for (in_chunk, out_chunk) in self.input_buffers[chan]
                    .chunks(self.fft_size_in)
                    .take(nbr_chunks_ready)
                    .zip(wave_out[chan].as_mut().chunks_mut(self.fft_size_out))
                {
                    self.resampler
                        .resample_unit(in_chunk, out_chunk, &mut self.overlaps[chan]);
                }
            }
        }

        // Save extra frames for next round.
        let frames_in_used = nbr_chunks_ready * self.fft_size_in;
        let extra = self.saved_frames - frames_in_used;

        if self.saved_frames > frames_in_used {
            for (chan, active) in self.channel_mask.iter().enumerate() {
                if *active {
                    self.input_buffers[chan].copy_within(frames_in_used..self.saved_frames, 0);
                }
            }
        }
        self.saved_frames = extra;
        Ok((self.chunk_size_in, needed_len))
    }

    fn input_frames_max(&self) -> usize {
        self.chunk_size_in
    }

    fn input_frames_next(&self) -> usize {
        self.chunk_size_in
    }

    fn nbr_channels(&self) -> usize {
        self.nbr_channels
    }

    fn output_frames_max(&self) -> usize {
        let max_stored_frames = self.fft_size_in - 1;
        let max_available_frames = max_stored_frames + self.chunk_size_in;
        let max_subchunks_to_process = max_available_frames / self.fft_size_in;
        max_subchunks_to_process * self.fft_size_out
    }

    fn output_frames_next(&self) -> usize {
        (((self.saved_frames + self.chunk_size_in) as f32) / self.fft_size_in as f32).floor()
            as usize
            * self.fft_size_out
    }

    fn output_delay(&self) -> usize {
        self.fft_size_out / 2
    }

    fn reset(&mut self) {
        self.overlaps
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = T::zero()));
        self.input_buffers
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = T::zero()));
        self.channel_mask.iter_mut().for_each(|val| *val = true);
        self.saved_frames = 0;
    }
}

/// Different window functions that can be used to window the sinc function.
#[derive(Debug, Clone, Copy)]
pub enum WindowFunction {
    /// Blackman. Intermediate rolloff and intermediate attenuation.
    Blackman,
    /// Squared Blackman. Slower rolloff but better attenuation than Blackman.
    Blackman2,
    /// Blackman-Harris. Slow rolloff but good attenuation.
    BlackmanHarris,
    /// Squared Blackman-Harris. Slower rolloff but better attenuation than Blackman-Harris.
    BlackmanHarris2,
    /// Hann. Fast rolloff but not very high attenuation.
    Hann,
    /// Squared Hann. Slower rolloff and higher attenuation than simple Hann.
    Hann2,
}

/// Helper function. Standard Blackman-Harris window.
// The window created is periodic.
pub fn blackman_harris<T>(npoints: usize) -> Vec<T>
where
    T: Float,
{
    let mut window = vec![T::zero(); npoints];
    let pi2 = T::from(2.0).unwrap() * T::from(std::f64::consts::PI).unwrap();
    let pi4 = T::from(4.0).unwrap() * T::from(std::f64::consts::PI).unwrap();
    let pi6 = T::from(6.0).unwrap() * T::from(std::f64::consts::PI).unwrap();
    let np_f = T::from(npoints).unwrap();
    let a = T::from(0.35875).unwrap();
    let b = T::from(0.48829).unwrap();
    let c = T::from(0.14128).unwrap();
    let d = T::from(0.01168).unwrap();
    for (x, item) in window.iter_mut().enumerate() {
        let x_float = T::from(x).unwrap();
        *item = a - b * (pi2 * x_float / np_f).cos() + c * (pi4 * x_float / np_f).cos()
            - d * (pi6 * x_float / np_f).cos();
    }
    window
}

/// Helper function. Standard Blackman window.
// The window created is periodic.
pub fn blackman<T>(npoints: usize) -> Vec<T>
where
    T: Float,
{
    let mut window = vec![T::zero(); npoints];
    let pi2 = T::from(2.0).unwrap() * T::from(std::f64::consts::PI).unwrap();
    let pi4 = T::from(4.0).unwrap() * T::from(std::f64::consts::PI).unwrap();
    let np_f = T::from(npoints).unwrap();
    let a = T::from(0.42).unwrap();
    let b = T::from(0.5).unwrap();
    let c = T::from(0.08).unwrap();
    for (x, item) in window.iter_mut().enumerate() {
        let x_float = T::from(x).unwrap();
        *item = a - b * (pi2 * x_float / np_f).cos() + c * (pi4 * x_float / np_f).cos();
    }
    window
}

/// Helper function. Standard Hann window.
// The window created is periodic.
pub fn hann<T>(npoints: usize) -> Vec<T>
where
    T: Float,
{
    // trace!("Making a Hann windows with {} points", npoints);
    let mut window = vec![T::zero(); npoints];
    let pi2 = T::from(2.0).unwrap() * T::from(std::f64::consts::PI).unwrap();
    let np_f = T::from(npoints).unwrap();
    let a = T::from(0.5).unwrap();
    for (x, item) in window.iter_mut().enumerate() {
        let x_float = T::from(x).unwrap();
        *item = a - a * (pi2 * x_float / np_f).cos();
    }
    window
}

/// Make the selected window function.
pub fn make_window<T>(npoints: usize, windowfunc: WindowFunction) -> Vec<T>
where
    T: Float,
{
    let mut window = match windowfunc {
        WindowFunction::BlackmanHarris | WindowFunction::BlackmanHarris2 => {
            blackman_harris::<T>(npoints)
        }
        WindowFunction::Blackman | WindowFunction::Blackman2 => blackman::<T>(npoints),
        WindowFunction::Hann | WindowFunction::Hann2 => hann::<T>(npoints),
    };
    match windowfunc {
        WindowFunction::Blackman2 | WindowFunction::BlackmanHarris2 | WindowFunction::Hann2 => {
            window.iter_mut().for_each(|y| *y = *y * *y);
        }
        _ => {}
    };
    window
}

/// Calculate a suitable relative cutoff frequency for the given sinc length using the given window function.
/// The result is based on an approximation, which gives good results for sinc lengths from 32 to 2048.
pub fn calculate_cutoff<T>(npoints: usize, windowfunc: WindowFunction) -> T
where
    T: Float,
{
    // Coefficient values generated by cutoff_fit_cubic.py
    let (k1, k2, k3) = match windowfunc {
        WindowFunction::BlackmanHarris => (
            T::from(8.041443677716476).unwrap(),
            T::from(55.9506779343387).unwrap(),
            T::from(898.0287985384213).unwrap(),
        ),
        WindowFunction::BlackmanHarris2 => (
            T::from(13.745202940783823).unwrap(),
            T::from(121.73532586374934).unwrap(),
            T::from(5964.163279612051).unwrap(),
        ),
        WindowFunction::Blackman => (
            T::from(6.159598046201173).unwrap(),
            T::from(18.926415097606878).unwrap(),
            T::from(653.4247430458968).unwrap(),
        ),
        WindowFunction::Blackman2 => (
            T::from(9.506235102129398).unwrap(),
            T::from(79.13120634953742).unwrap(),
            T::from(1502.2316160588925).unwrap(),
        ),
        WindowFunction::Hann => (
            T::from(3.3481080887677166).unwrap(),
            T::from(10.106519434875038).unwrap(),
            T::from(78.96345249024414).unwrap(),
        ),
        WindowFunction::Hann2 => (
            T::from(5.38751148378734).unwrap(),
            T::from(29.69451915489501).unwrap(),
            T::from(184.82117462266237).unwrap(),
        ),
    };
    let one = T::one();
    one / (k1 / T::from(npoints).unwrap()
        + k2 / T::from(npoints.pow(2)).unwrap()
        + k3 / T::from(npoints.pow(3)).unwrap()
        + one)
}

pub(crate) fn validate_buffers<T, Vin: AsRef<[T]>, Vout: AsMut<[T]>>(
    wave_in: &[Vin],
    wave_out: &mut [Vout],
    mask: &[bool],
    channels: usize,
    min_input_len: usize,
    min_output_len: usize,
) -> Result<(), &'static str> {
    if wave_in.len() != channels {
        return Err("ResampleError::WrongNumberOfInputChannels");
    }
    if mask.len() != channels {
        return Err("ResampleError::WrongNumberOfMaskChannels");
    }
    for (chan, wave_in) in wave_in.iter().enumerate().filter(|(chan, _)| mask[*chan]) {
        let actual_len = wave_in.as_ref().len();
        if actual_len < min_input_len {
            return Err("ResampleError::InsufficientInputBufferSize");
        }
    }
    if wave_out.len() != channels {
        return Err("ResampleError::WrongNumberOfOutputChannels");
    }
    for (chan, wave_out) in wave_out
        .iter_mut()
        .enumerate()
        .filter(|(chan, _)| mask[*chan])
    {
        let actual_len = wave_out.as_mut().len();
        if actual_len < min_output_len {
            return Err("ResampleError::InsufficientOutputBufferSize");
        }
    }
    Ok(())
}

/// Helper to make a mask where all channels are marked as active.
fn update_mask_from_buffers(mask: &mut [bool]) {
    mask.iter_mut().for_each(|v| *v = true);
}

/// Helper function: sinc(x) = sin(pi*x)/(pi*x).
pub fn sinc<T>(value: T) -> T
where
    T: Float,
{
    if value == T::zero() {
        T::one()
    } else {
        (value * T::from(std::f64::consts::PI).unwrap()).sin()
            / (value * T::from(std::f64::consts::PI).unwrap())
    }
}

/// Helper function. Make a set of windowed sincs.
pub fn make_sincs<T>(
    npoints: usize,
    factor: usize,
    f_cutoff: f32,
    windowfunc: WindowFunction,
) -> Vec<Vec<T>>
where
    T: Float,
{
    let totpoints = npoints * factor;
    let mut y = Vec::with_capacity(totpoints);
    let window = make_window::<T>(totpoints, windowfunc);
    let mut sum = T::zero();
    for (x, w) in window.iter().enumerate().take(totpoints) {
        let val = *w
            * sinc(
                (T::from(x).unwrap() - T::from(totpoints / 2).unwrap())
                    * T::from(f_cutoff).unwrap()
                    / T::from(factor).unwrap(),
            );
        sum = sum + val;
        y.push(val);
    }
    sum = sum / T::from(factor).unwrap();
    // debug!(
    //     "Generate sincs, length: {}, oversampling: {}, normalized by: {:?}",
    //     npoints, factor, sum
    // );
    let mut sincs = vec![vec![T::zero(); npoints]; factor];
    for p in 0..npoints {
        for n in 0..factor {
            sincs[factor - n - 1][p] = y[factor * p + n] / sum;
        }
    }
    sincs
}
