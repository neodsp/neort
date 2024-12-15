use num::Float;

use crate::audio_block::BlockWrite;

#[allow(unused)]
pub struct AudioSettings {
    pub sample_rate: f64,
    pub num_channels: u16,
    pub max_num_frames: u32,
}

pub trait Processor<F: Float + 'static> {
    type Result;
    type Parameter;

    /// This function will be called from another thread!
    /// To change a value that is accessed by the process function,
    /// always use thread synchronization primitives like channels or atomics!
    fn set_parameter(&mut self, param: Self::Parameter) -> Self::Result;

    /// This always needs to be called before the processing start.
    /// Here you can still allocate and block, but don't block for too long,
    /// or your processor with block the startup of the whole audio app.
    fn prepare(&mut self, settings: &AudioSettings) -> Self::Result;

    /// This is meant to be called in a "real-time" audio thread.
    /// Do nothing expensive and nothing blocking here.
    /// It is recommended to sanitize this function with the rtsan crate.
    fn process(&mut self, block: &mut impl BlockWrite<F>) -> Self::Result;

    /// This can be used to reset states in the processor, like emptying a delay.
    fn reset(&mut self);
}
