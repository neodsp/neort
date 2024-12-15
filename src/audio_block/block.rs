use ndarray::{
    iter::{Lanes, LanesMut},
    s, Array2, ArrayView1, ArrayViewMut1, Dim,
};
use num::Float;

use super::{block_view::BlockView, block_view_mut::BlockViewMut, BlockRead, BlockWrite};

#[derive(Debug, Clone, PartialEq)]
pub struct Block<F: Float> {
    data: Array2<F>,
}

impl<F: Float> Default for Block<F> {
    fn default() -> Self {
        Self {
            data: Array2::zeros((0, 0)),
        }
    }
}

impl<F: Float> Block<F> {
    pub fn new(num_channels: u16, num_frames: usize) -> Self {
        Self {
            data: Array2::zeros((num_channels as usize, num_frames)),
        }
    }

    pub fn from_array(array: Array2<F>) -> Self {
        Self { data: array }
    }
}

impl<F: Float + 'static> BlockRead<F> for Block<F> {
    #[rtsan::nonblocking]
    #[inline(always)]
    fn channel(&self, index: u16) -> ArrayView1<F> {
        self.data.row(index as usize)
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn frame(&self, index: usize) -> ArrayView1<F> {
        self.data.column(index)
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn channels(&self) -> impl Iterator<Item = ArrayView1<F>> {
        self.data.rows().into_iter()
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn frames(&self) -> Lanes<F, Dim<[usize; 1]>> {
        self.data.columns()
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn view(&self) -> BlockView<F> {
        BlockView::from_array_view(self.data.view())
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn view_slice(&self, range: std::ops::Range<usize>) -> BlockView<F> {
        BlockView::from_array_view(self.data.slice(s![.., range]))
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn raw_buffer(&self) -> &[F] {
        self.data.as_slice_memory_order().unwrap()
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn data(&self) -> ndarray::ArrayView2<F> {
        self.data.view()
    }
}

impl<F: Float + 'static> BlockWrite<F> for Block<F> {
    #[rtsan::nonblocking]
    #[inline(always)]
    fn sample_mut(&mut self, ch: u16, frame: usize) -> &mut F {
        &mut self.data[[ch as usize, frame]]
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn channel_mut(&mut self, index: u16) -> ArrayViewMut1<F> {
        self.data.row_mut(index as usize)
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn frame_mut(&mut self, index: usize) -> ArrayViewMut1<F> {
        self.data.column_mut(index)
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn channels_mut(&mut self) -> impl Iterator<Item = ArrayViewMut1<F>> {
        self.data.rows_mut().into_iter()
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn frames_mut(&mut self) -> LanesMut<F, Dim<[usize; 1]>> {
        self.data.columns_mut()
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn raw_buffer_mut(&mut self) -> &mut [F] {
        self.data.as_slice_memory_order_mut().unwrap()
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn view_mut(&mut self) -> BlockViewMut<F> {
        BlockViewMut::from_array_view(self.data.view_mut())
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn view_slice_mut(&mut self, range: std::ops::Range<usize>) -> BlockViewMut<F> {
        BlockViewMut::from_array_view(self.data.slice_mut(s![.., range]))
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn data_mut(&mut self) -> ndarray::ArrayViewMut2<F> {
        self.data.view_mut()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, aview1, aview2, aview_mut2};

    use crate::audio_block::BufferLayout;

    use super::*;

    #[test]
    fn block() {
        let block = Block::<f32>::new(2, 3);
        assert_eq!(
            block.view(),
            BlockView::from_array_view(aview2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
        );

        let mut block = Block::<f32>::from_array(array![[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]);

        // Block Read
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.layout(), BufferLayout::Sequential);

        // sample
        assert_eq!(block.sample(0, 1), 0.1);
        assert_eq!(block.sample(1, 2), 1.2);

        // fn channel(&self, index: u16) -> ArrayView1<Sample>;
        assert_eq!(block.channel(0), aview1(&[0.0, 0.1, 0.2]));
        assert_eq!(block.channel(1), aview1(&[1.0, 1.1, 1.2]));
        // fn frame(&self, index: u32) -> ArrayView1<Sample>;
        assert_eq!(block.frame(0), aview1(&[0.0, 1.0]));
        assert_eq!(block.frame(1), aview1(&[0.1, 1.1]));
        assert_eq!(block.frame(2), aview1(&[0.2, 1.2]));
        // fn channels(&self) -> Lanes<Sample, Dim<[usize; 1]>>;
        for (ch, channel) in block.channels().enumerate() {
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
            BlockView::from_array_view(aview2(&[[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]))
        );
        // fn raw_buffer(&self) -> &[Sample];
        assert_eq!(block.raw_buffer(), &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2]);

        // Block write
        // sample_mut
        assert_eq!(*block.sample_mut(0, 1), 0.1);
        assert_eq!(*block.sample_mut(1, 2), 1.2);
        // fn channel_mut(&mut self, index: u16) -> ArrayViewMut1<Sample>;
        assert_eq!(block.channel_mut(0), aview1(&[0.0, 0.1, 0.2]));
        assert_eq!(block.channel_mut(1), aview1(&[1.0, 1.1, 1.2]));
        // fn frame_mut(&mut self, index: u32) -> ArrayViewMut1<Sample>;
        assert_eq!(block.frame_mut(0), aview1(&[0.0, 1.0]));
        assert_eq!(block.frame_mut(1), aview1(&[0.1, 1.1]));
        assert_eq!(block.frame_mut(2), aview1(&[0.2, 1.2]));
        // fn channels_mut(&mut self) -> LanesMut<Sample, Dim<[usize; 1]>>;
        for (ch, channel) in block.channels_mut().into_iter().enumerate() {
            if ch == 0 {
                assert_eq!(channel, aview1(&[0.0, 0.1, 0.2]));
            } else if ch == 1 {
                assert_eq!(channel, aview1(&[1.0, 1.1, 1.2]));
            }
        }
        // fn frames_mut(&mut self) -> LanesMut<Sample, Dim<[usize; 1]>>;
        for (fr, frame) in block.frames_mut().into_iter().enumerate() {
            if fr == 0 {
                assert_eq!(frame, aview1(&[0.0, 1.0]));
            } else if fr == 1 {
                assert_eq!(frame, aview1(&[0.1, 1.1]));
            } else if fr == 2 {
                assert_eq!(frame, aview1(&[0.2, 1.2]));
            }
        }
        // fn view_mut(&mut self) -> BlockViewMut<Sample>;
        assert_eq!(
            block.view_mut(),
            BlockViewMut::from_array_view(aview_mut2(&mut [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]),)
        );
        // fn raw_buffer_mut(&mut self) -> &mut [Sample];
        assert_eq!(block.raw_buffer_mut(), &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2]);
    }
}
