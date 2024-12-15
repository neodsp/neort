use ndarray::{
    iter::{Lanes, LanesMut},
    s, ArrayView1, ArrayViewMut1, ArrayViewMut2, Dim, ShapeBuilder,
};
use num::Float;
use rtsan::{blocking, nonblocking};

use super::{block_view::BlockView, Block, BlockRead, BlockWrite, BufferLayout};

#[derive(Debug, PartialEq)]
pub struct BlockViewMut<'a, F: Float> {
    data: ArrayViewMut2<'a, F>,
}

impl<F: Float> Default for BlockViewMut<'_, F> {
    #[nonblocking]
    fn default() -> Self {
        Self {
            data: ArrayViewMut2::from_shape((0, 0), &mut []).unwrap(),
        }
    }
}

impl<'a, F: Float> BlockViewMut<'a, F> {
    #[nonblocking]
    pub fn from_buffer(
        buffer: &'a mut [F],
        num_channels: u16,
        num_frames: usize,
        layout: BufferLayout,
    ) -> Self {
        let data = match layout {
            BufferLayout::Sequential => {
                ArrayViewMut2::from_shape((num_channels as usize, num_frames), buffer).unwrap()
            }
            BufferLayout::Interleaved => {
                ArrayViewMut2::from_shape((num_channels as usize, num_frames).f(), buffer).unwrap()
            }
        };
        Self { data }
    }

    #[nonblocking]
    pub fn from_array_view(view: ArrayViewMut2<'a, F>) -> Self {
        Self { data: view }
    }

    // TODO: Test
    /// # Safety
    /// The buffer the pointer points to must be at least `num_channels` * `num_frames`
    /// elements long. Otherwise this is undefined behavior.
    #[nonblocking]
    #[inline(always)]
    pub unsafe fn from_ptr(ptr: *mut F, num_channels: u16, num_frames: usize) -> Self {
        Self {
            data: ArrayViewMut2::from_shape_ptr((num_channels as usize, num_frames), ptr),
        }
    }

    #[blocking]
    pub fn to_owned(&self, force_sequential_layout: bool) -> Block<F> {
        if force_sequential_layout {
            Block::from_array(self.data.as_standard_layout().to_owned())
        } else {
            Block::from_array(self.data.to_owned())
        }
    }
}

impl<F: Float> BlockRead<F> for BlockViewMut<'_, F> {
    #[nonblocking]
    #[inline(always)]
    fn channel(&self, index: u16) -> ArrayView1<F> {
        self.data.row(index as usize)
    }

    #[nonblocking]
    #[inline(always)]
    fn frame(&self, index: usize) -> ArrayView1<F> {
        self.data.column(index)
    }

    #[nonblocking]
    #[inline(always)]
    fn channels(&self) -> Lanes<F, Dim<[usize; 1]>> {
        self.data.rows()
    }

    #[nonblocking]
    #[inline(always)]
    fn frames(&self) -> Lanes<F, Dim<[usize; 1]>> {
        self.data.columns()
    }

    #[nonblocking]
    #[inline(always)]
    fn view(&self) -> BlockView<F> {
        BlockView::from_array_view(self.data.view())
    }

    #[nonblocking]
    #[inline(always)]
    fn view_slice(&self, range: std::ops::Range<usize>) -> BlockView<F> {
        BlockView::from_array_view(self.data.slice(s![.., range]))
    }

    #[nonblocking]
    #[inline(always)]
    fn raw_buffer(&self) -> &[F] {
        self.data.as_slice_memory_order().unwrap()
    }

    #[nonblocking]
    #[inline(always)]
    fn data(&self) -> ndarray::ArrayView2<F> {
        self.data.view()
    }
}

impl<F: Float> BlockWrite<F> for BlockViewMut<'_, F> {
    #[nonblocking]
    #[inline(always)]
    fn sample_mut(&mut self, ch: u16, frame: usize) -> &mut F {
        &mut self.data[[ch as usize, frame]]
    }

    #[nonblocking]
    #[inline(always)]
    fn channel_mut(&mut self, index: u16) -> ArrayViewMut1<F> {
        self.data.row_mut(index as usize)
    }

    #[nonblocking]
    #[inline(always)]
    fn frame_mut(&mut self, index: usize) -> ArrayViewMut1<F> {
        self.data.column_mut(index)
    }

    #[nonblocking]
    #[inline(always)]
    fn channels_mut(&mut self) -> LanesMut<F, Dim<[usize; 1]>> {
        self.data.rows_mut()
    }

    #[nonblocking]
    #[inline(always)]
    fn frames_mut(&mut self) -> LanesMut<F, Dim<[usize; 1]>> {
        self.data.columns_mut()
    }

    #[nonblocking]
    #[inline(always)]
    fn view_mut(&mut self) -> BlockViewMut<F> {
        BlockViewMut::from_array_view(self.data.view_mut())
    }

    #[nonblocking]
    #[inline(always)]
    fn view_slice_mut(&mut self, range: std::ops::Range<usize>) -> BlockViewMut<F> {
        BlockViewMut::from_array_view(self.data.slice_mut(s![.., range]))
    }

    #[nonblocking]
    #[inline(always)]
    fn raw_buffer_mut(&mut self) -> &mut [F] {
        self.data.as_slice_memory_order_mut().unwrap()
    }

    #[nonblocking]
    #[inline(always)]
    fn data_mut(&mut self) -> ndarray::ArrayViewMut2<F> {
        self.data.view_mut()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, aview1, aview2};

    use super::*;

    // TODO: Mut access tests missing
    #[test]
    fn block_view_mut_sequential() {
        let mut buffer = [0.0, 0.1, 0.2, 1.0, 1.1, 1.2];
        let block = BlockViewMut::from_buffer(&mut buffer, 2, 3, BufferLayout::Sequential);

        assert_eq!(
            block.to_owned(false),
            Block::from_array(array![[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]])
        );

        assert_eq!(
            block.to_owned(false).raw_buffer(),
            &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2]
        );

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
            BlockView::from_array_view(aview2(&[[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]))
        );
        // fn raw_buffer(&self) -> &[Sample];
        assert_eq!(block.raw_buffer(), &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2]);
    }

    #[test]
    fn block_view_mut_interleaved() {
        let mut buffer = [0.0, 1.0, 0.1, 1.1, 0.2, 1.2];
        let block = BlockViewMut::from_buffer(&mut buffer, 2, 3, BufferLayout::Interleaved);

        // sample
        assert_eq!(block.sample(0, 1), 0.1);
        assert_eq!(block.sample(1, 2), 1.2);

        assert_eq!(
            block.to_owned(false),
            Block::from_array(array![[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]])
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
            BlockView::from_array_view(aview2(&[[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]))
        );
        assert_eq!(block.raw_buffer(), &[0.0, 1.0, 0.1, 1.1, 0.2, 1.2]);
    }
}
