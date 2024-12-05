use ndarray::{iter::Lanes, ArrayView1, ArrayView2, Dim, ShapeBuilder};
use num::Float;

use super::{Block, BlockRead, BufferLayout};

#[derive(Debug, Clone, PartialEq)]
pub struct BlockView<'a, Sample: Float> {
    data: ArrayView2<'a, Sample>,
    sample_rate: f64,
}

impl<'a, Sample: Float> BlockView<'a, Sample> {
    #[rtsan::nonblocking]
    pub fn from_buffer(
        buffer: &'a [Sample],
        sample_rate: f64,
        num_channels: u16,
        num_frames: usize,
        layout: BufferLayout,
    ) -> Self {
        let data = match layout {
            BufferLayout::Sequential => {
                ArrayView2::from_shape((num_channels as usize, num_frames), buffer)
                    .unwrap()
            }
            BufferLayout::Interleaved => {
                ArrayView2::from_shape((num_channels as usize, num_frames).f(), buffer)
                    .unwrap()
            }
        };
        Self { data, sample_rate }
    }

    #[rtsan::nonblocking]
    pub fn from_array_view(view: ArrayView2<'a, Sample>, sample_rate: f64) -> Self {
        Self {
            data: view,
            sample_rate,
        }
    }

    #[rtsan::nonblocking]
    pub fn layout(&self) -> BufferLayout {
        if self.data.is_standard_layout() {
            BufferLayout::Sequential
        } else {
            BufferLayout::Interleaved
        }
    }

    pub fn to_owned(&self, force_sequential_layout: bool) -> Block<Sample> {
        if force_sequential_layout {
            Block::from_array(self.data.as_standard_layout().to_owned(), self.sample_rate)
        } else {
            Block::from_array(self.data.to_owned(), self.sample_rate)
        }
    }
}

impl<Sample: Float> BlockRead<Sample> for BlockView<'_, Sample> {
    #[rtsan::nonblocking]
    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    #[rtsan::nonblocking]
    fn num_channels(&self) -> u16 {
        self.data.nrows() as u16
    }

    #[rtsan::nonblocking]
    fn num_frames(&self) -> u32 {
        self.data.ncols() as u32
    }

    #[rtsan::nonblocking]
    fn layout(&self) -> BufferLayout {
        if self.data.is_standard_layout() {
            BufferLayout::Sequential
        } else {
            BufferLayout::Interleaved
        }
    }

    #[rtsan::nonblocking]
    fn channel(&self, index: u16) -> ArrayView1<Sample> {
        self.data.row(index as usize)
    }

    #[rtsan::nonblocking]
    fn frame(&self, index: u32) -> ArrayView1<Sample> {
        self.data.column(index as usize)
    }

    #[rtsan::nonblocking]
    fn channels(&self) -> Lanes<Sample, Dim<[usize; 1]>> {
        self.data.rows()
    }

    #[rtsan::nonblocking]
    fn frames(&self) -> Lanes<Sample, Dim<[usize; 1]>> {
        self.data.columns()
    }

    #[rtsan::nonblocking]
    fn raw_buffer(&self) -> &[Sample] {
        self.data.as_slice_memory_order().unwrap()
    }

    #[rtsan::nonblocking]
    fn view(&self) -> BlockView<Sample> {
        BlockView::from_array_view(self.data.view(), self.sample_rate)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, aview1, aview2};

    use super::*;

    #[test]
    fn block_view_sequential() {
        let block = BlockView::from_buffer(
            &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2],
            44100.0,
            2,
            3,
            BufferLayout::Sequential,
        );

        assert_eq!(
            block.to_owned(false),
            Block::from_array(array![[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], 44100.0)
        );

        assert_eq!(
            block.to_owned(false).raw_buffer(),
            &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2]
        );

        // Block Read
        assert_eq!(block.sample_rate(), 44100.0);
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.layout(), BufferLayout::Sequential);

        // fn channel(&self, index: u16) -> ArrayView1<Sample>;
        assert_eq!(block.channel(0), aview1(&[0.0, 0.1, 0.2]));
        assert_eq!(block.channel(1), aview1(&[1.0, 1.1, 1.2]));
        // fn frame(&self, index: u32) -> ArrayView1<Sample>;
        assert_eq!(block.frame(0), aview1(&[0.0, 1.0]));
        assert_eq!(block.frame(1), aview1(&[0.1, 1.1]));
        assert_eq!(block.frame(2), aview1(&[0.2, 1.2]));
        // fn channels(&self) -> Lanes<Sample, Dim<[usize; 1]>>;
        for (ch, channel) in block.channels().into_iter().enumerate() {
            if ch == 0 {
                assert_eq!(channel, aview1(&[0.0, 0.1, 0.2]));
            } else if ch == 1 {
                assert_eq!(channel, aview1(&[1.0, 1.1, 1.2]));
            }
        }
        // fn frames(&self) -> Lanes<Sample, Dim<[usize; 1]>>;
        for (fr, frame) in block.frames().into_iter().enumerate() {
            if fr == 0 {
                assert_eq!(frame, aview1(&[0.0, 1.0]));
            } else if fr == 1 {
                assert_eq!(frame, aview1(&[0.1, 1.1]));
            } else if fr == 2 {
                assert_eq!(frame, aview1(&[0.2, 1.2]));
            }
        }
        // fn view(&self) -> BlockView<Sample>;
        assert_eq!(
            block.view(),
            BlockView::from_array_view(aview2(&[[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]), 44100.0)
        );
        // fn raw_buffer(&self) -> &[Sample];
        assert_eq!(block.raw_buffer(), &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2]);
    }

    #[test]
    fn block_view_interleaved() {
        let block = BlockView::from_buffer(
            &[0.0, 1.0, 0.1, 1.1, 0.2, 1.2],
            44100.0,
            2,
            3,
            BufferLayout::Interleaved,
        );

        assert_eq!(
            block.to_owned(false),
            Block::from_array(array![[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], 44100.0)
        );

        assert_eq!(
            block.to_owned(false).raw_buffer(),
            &[0.0, 1.0, 0.1, 1.1, 0.2, 1.2]
        );

        assert_eq!(
            block.to_owned(true).raw_buffer(),
            &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2]
        );

        assert_eq!(block.layout(), BufferLayout::Interleaved);
        assert_eq!(
            block.view(),
            BlockView::from_array_view(aview2(&[[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]), 44100.0)
        );
        assert_eq!(block.raw_buffer(), &[0.0, 1.0, 0.1, 1.1, 0.2, 1.2]);
    }
}
