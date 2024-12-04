use num::Float;

use super::{AudioBlock, AudioBlockMut};

#[derive(Debug)]
pub struct SequentialBlock<Sample: Float + 'static> {
    data: Vec<Sample>,
    sample_rate: f64,
    num_channels: u16,
    num_frames: u32,
}

impl<Sample: Float + 'static> SequentialBlock<Sample> {
    pub fn new(sample_rate: f64, num_channels: u16, num_frames: u32) -> Self {
        assert!(sample_rate > 0.0);
        Self {
            data: vec![Sample::zero(); num_channels as usize * num_frames as usize],
            sample_rate,
            num_channels,
            num_frames,
        }
    }

    pub fn resize(&mut self, num_channels: u16, num_frames: u32) {
        self.data
            .resize(num_channels as usize * num_frames as usize, Sample::zero());
    }
}

impl<Sample: Float + 'static> AudioBlock<Sample> for SequentialBlock<Sample> {
    type ChannelIter<'s> = std::iter::Take<std::iter::Skip<std::slice::Iter<'s, Sample>>>;
    type FrameIter<'s> = std::iter::StepBy<std::iter::Skip<std::slice::Iter<'s, Sample>>>;

    #[rtsan::nonblocking]
    #[inline(always)]
    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn num_channels(&self) -> u16 {
        self.num_channels
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn num_frames(&self) -> u32 {
        self.num_frames
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn length_in_seconds(&self) -> f64 {
        self.num_frames as f64 / self.sample_rate
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn sample(&self, channel: u16, frame: u32) -> Sample {
        self.data[frame as usize + self.num_frames as usize * channel as usize]
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn channel<'s>(&'s self, channel: u16) -> Self::ChannelIter<'s> {
        self.data
            .iter()
            .skip(self.num_frames as usize * channel as usize)
            .take(self.num_frames as usize)
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn frame<'s>(&'s self, frame: u32) -> Self::FrameIter<'s> {
        self.data
            .iter()
            .skip(frame as usize)
            .step_by(self.num_frames as usize)
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn raw_buffer(&self) -> &[Sample] {
        &self.data
    }
}

impl<Sample: Float + 'static> AudioBlockMut<Sample> for SequentialBlock<Sample> {
    type ChannelMutIter<'s> = std::iter::Take<std::iter::Skip<std::slice::IterMut<'s, Sample>>>;
    type FrameMutIter<'s> = std::iter::StepBy<std::iter::Skip<std::slice::IterMut<'s, Sample>>>;

    #[rtsan::nonblocking]
    #[inline(always)]
    fn sample_mut(&mut self, channel: u16, frame: u32) -> &mut Sample {
        &mut self.data[frame as usize + self.num_frames as usize * channel as usize]
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn channel_mut<'s>(&'s mut self, channel: u16) -> Self::ChannelMutIter<'s> {
        self.data
            .iter_mut()
            .skip(self.num_frames as usize * channel as usize)
            .take(self.num_frames as usize)
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn frame_mut<'s>(&'s mut self, frame: u32) -> Self::FrameMutIter<'s> {
        self.data
            .iter_mut()
            .skip(frame as usize)
            .step_by(self.num_frames as usize)
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn raw_buffer_mut(&mut self) -> &mut [Sample] {
        &mut self.data
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_block() {
        let mut block = SequentialBlock::<f32>::new(8.0, 2, 8);

        assert_eq!(block.sample_rate(), 8.0);
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 8);
        assert_eq!(block.length_in_seconds(), 1.0);

        // Channels
        block.channel_mut(0).for_each(|s| *s = 1.0);
        block.channel_mut(1).for_each(|s| *s = 2.0);

        assert_eq!(
            block.raw_buffer(),
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        );
        assert_eq!(
            block.channel(0).copied().collect::<Vec<_>>(),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
        assert_eq!(
            block.channel(1).copied().collect::<Vec<_>>(),
            vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        );

        // Frames
        block.frame_mut(2).for_each(|s| *s = 10.0);

        assert_eq!(
            block.raw_buffer(),
            &[1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 10.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        );
        assert_eq!(block.frame(1).copied().collect::<Vec<_>>(), vec![1.0, 2.0]);
        assert_eq!(
            block.frame(2).copied().collect::<Vec<_>>(),
            vec![10.0, 10.0]
        );

        // Raw Buffer
        block.raw_buffer_mut()[10] = 20.0;

        assert_eq!(
            block.raw_buffer(),
            &[1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 20.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        );

        // Sample
        *block.sample_mut(1, 4) = 30.0;
        assert_eq!(
            block.raw_buffer(),
            &[1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 20.0, 2.0, 30.0, 2.0, 2.0, 2.0]
        );
        assert_eq!(block.sample(1, 4), 30.0);
        assert_eq!(block.sample(1, 7), 2.0);
    }
}
