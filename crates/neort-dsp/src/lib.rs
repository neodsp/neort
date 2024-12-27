pub mod adapter;
pub mod resampler;
pub mod ringbuffer;

pub trait Float: num::Float + realfft::FftNum {}
impl<T> Float for T where T: num::Float + realfft::FftNum {}
