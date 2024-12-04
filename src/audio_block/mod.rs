use num::Float;

mod interleaved;

pub trait AudioBlock<Sample: Float + 'static> {
    type ChannelIter<'s>: Iterator<Item = &'s Sample>
    where
        Self: 's;
    type FrameIter<'s>: Iterator<Item = &'s Sample>
    where
        Self: 's;

    fn sample_rate(&self) -> f64;
    fn num_channels(&self) -> u16;
    fn num_frames(&self) -> u32;
    fn length_in_seconds(&self) -> f64;

    fn sample(&self, channel: u16, frame: u32) -> Sample;
    fn channel<'s>(&'s self, channel: u16) -> Self::ChannelIter<'s>;
    fn frame<'s>(&'s self, frame: u32) -> Self::FrameIter<'s>;
    fn raw_buffer(&self) -> &[Sample];
}

pub trait AudioBlockMut<Sample: Float + 'static>: AudioBlock<Sample> {
    type ChannelMutIter<'s>: Iterator<Item = &'s mut Sample>
    where
        Self: 's;
    type FrameMutIter<'s>: Iterator<Item = &'s mut Sample>
    where
        Self: 's;

    fn sample_mut(&mut self, channel: u16, frame: u32) -> &mut Sample;
    fn channel_mut<'s>(&'s mut self, channel: u16) -> Self::ChannelMutIter<'s>;
    fn frame_mut<'s>(&'s mut self, frame: u32) -> Self::FrameMutIter<'s>;
    fn raw_buffer_mut(&mut self) -> &mut [Sample];
}
