pub mod adapter;
pub mod resampler;
pub mod ringbuffer;

pub trait Float: num_traits::Float + realfft::FftNum {}
impl<T> Float for T where T: num_traits::Float + realfft::FftNum {}
