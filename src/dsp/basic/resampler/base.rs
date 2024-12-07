// The resamplers are copied from rubato by Henrik Enquist and adapted to take Blocks

use std::sync::Arc;

use ndarray::{ArrayView1, ArrayViewMut1};
use num::{Complex, Float, Zero};
use realfft::{ComplexToReal, FftNum, RealFftPlanner, RealToComplex};

use super::utils::{calculate_cutoff, make_sincs, WindowFunction};

/// A helper for resampling a single chunk of data.
pub struct FftResampler<F: Float + FftNum> {
    fft_size_in: usize,
    fft_size_out: usize,
    filter_f: Vec<Complex<F>>,
    fft: Arc<dyn RealToComplex<F>>,
    ifft: Arc<dyn ComplexToReal<F>>,
    scratch_fw: Vec<Complex<F>>,
    scratch_inv: Vec<Complex<F>>,
    input_buf: Vec<F>,
    input_f: Vec<Complex<F>>,
    output_f: Vec<Complex<F>>,
    output_buf: Vec<F>,
}

impl<F: Float + FftNum> FftResampler<F> {
    pub fn new(fft_size_in: usize, fft_size_out: usize) -> Self {
        // calculate antialiasing cutoff
        let cutoff = if fft_size_in > fft_size_out {
            calculate_cutoff::<f32>(fft_size_out, WindowFunction::BlackmanHarris2)
                * fft_size_out as f32
                / fft_size_in as f32
        } else {
            calculate_cutoff::<f32>(fft_size_in, WindowFunction::BlackmanHarris2)
        };
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
    #[rtsan::nonblocking]
    pub fn resample_unit(
        &mut self,
        input: ArrayView1<F>,
        mut output: ArrayViewMut1<F>,
        overlap: &mut [F],
    ) {
        // Copy to input buffer and clear padding area.
        self.input_buf
            .iter_mut()
            .take(self.fft_size_in)
            .zip(input.iter())
            .for_each(|(a, b)| *a = *b);
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
        for (n, item) in output.iter_mut().enumerate().take(self.fft_size_out) {
            *item = self.output_buf[n] + overlap[n];
        }
        overlap.copy_from_slice(&self.output_buf[self.fft_size_out..]);
    }
}
