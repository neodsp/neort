use rtsan_standalone::nonblocking;

use crate::{Block, BlockDataConst, BlockDataMut};

// Planar Copy Tools
impl<D: BlockDataConst> Block<D> {
    #[nonblocking]
    #[allow(clippy::missing_safety_doc)]
    pub fn copy_into_interleaved_data(
        &self,
        data: &mut [D::Num],
        num_channels: usize,
        num_frames: usize,
    ) {
        assert!(num_channels * num_frames == data.len());
        assert_eq!(self.num_channels, num_channels);
        assert_eq!(self.num_frames, num_frames);
        for channel in 0..num_channels {
            for (output, input) in data
                .iter_mut()
                .skip(channel)
                .step_by(num_channels)
                .zip(self.channel(channel).iter())
            {
                *output = *input;
            }
        }
    }
}

// Planar Copy Tools
impl<D: BlockDataMut> Block<D> {
    #[nonblocking]
    #[allow(clippy::missing_safety_doc)]
    pub fn copy_from_interleaved_data(
        &mut self,
        data: &[D::Num],
        num_channels: usize,
        num_frames: usize,
    ) {
        assert!(num_channels * num_frames == data.len());
        self.set_num_channels_visible(num_channels);
        self.set_num_frames_visible(num_frames);
        for channel in 0..num_channels {
            for (input, output) in data
                .iter()
                .skip(channel)
                .step_by(num_channels)
                .zip(self.channel_mut(channel).iter_mut())
            {
                *output = *input;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::BlockHeap;

    #[test]
    fn test_interleaved() {
        let interleaved = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9];
        let mut block = BlockHeap::<i32>::new(3, 6);
        block.copy_from_interleaved_data(&interleaved, 2, 5);
        assert_eq!(block[0], [0, 1, 2, 3, 4]);
        assert_eq!(block[1], [5, 6, 7, 8, 9]);

        let mut interleaved = [0; 10];
        block.copy_into_interleaved_data(&mut interleaved, 2, 5);
        assert_eq!(interleaved, [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]);
    }
}
