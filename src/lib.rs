use std::fmt::Debug;

use num::Float;
use realfft::FftNum;

pub mod audio_processor;
pub mod dsp;
pub mod ringbuffer;
// #[cfg(any(
//     feature = "system-audio-cubeb",
//     feature = "system-audio-juce",
//     feature = "system-audio-rtaudio"
// ))]
// pub mod system_audio;

pub trait Sample: Float + Debug + FftNum + 'static {}

impl Sample for f32 {}
impl Sample for f64 {}
