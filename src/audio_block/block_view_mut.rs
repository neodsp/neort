use ndarray::{
    iter::{Lanes, LanesMut},
    ArrayView1, ArrayViewMut1, ArrayViewMut2, Dim, ShapeBuilder,
};
use num::Float;

use super::{block_view::BlockView, Block, BlockRead, BlockWrite, BufferLayout};

pub struct BlockViewMut<'a, Sample: Float> {
    data: ArrayViewMut2<'a, Sample>,
    sample_rate: f64,
}

impl<'a, Sample: Float> BlockViewMut<'a, Sample> {
    #[rtsan::nonblocking]
    pub fn from_buffer(
        buffer: &'a mut [Sample],
        sample_rate: f64,
        num_channels: u16,
        num_frames: usize,
        layout: BufferLayout,
    ) -> Self {
        let data = match layout {
            BufferLayout::Sequential => {
                ArrayViewMut2::from_shape((num_channels as usize, num_frames as usize), buffer)
                    .unwrap()
            }
            BufferLayout::Interleaved => {
                ArrayViewMut2::from_shape((num_channels as usize, num_frames as usize).f(), buffer)
                    .unwrap()
            }
        };
        Self { data, sample_rate }
    }

    #[rtsan::nonblocking]
    pub fn from_array_view(view: ArrayViewMut2<'a, Sample>, sample_rate: f64) -> Self {
        Self {
            data: view,
            sample_rate: sample_rate,
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

    pub fn to_owned(&self) -> Block<Sample> {
        Block::from_array(self.data.to_owned(), self.sample_rate)
    }
}

impl<'a, Sample: Float> BlockRead<Sample> for BlockViewMut<'a, Sample> {
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

impl<'a, Sample: Float> BlockWrite<Sample> for BlockViewMut<'a, Sample> {
    #[rtsan::nonblocking]
    fn channel_mut(&mut self, index: u16) -> ArrayViewMut1<Sample> {
        self.data.row_mut(index as usize)
    }

    #[rtsan::nonblocking]
    fn frame_mut(&mut self, index: u32) -> ArrayViewMut1<Sample> {
        self.data.column_mut(index as usize)
    }

    #[rtsan::nonblocking]
    fn channels_mut(&mut self) -> LanesMut<Sample, Dim<[usize; 1]>> {
        self.data.rows_mut()
    }

    #[rtsan::nonblocking]
    fn frames_mut(&mut self) -> LanesMut<Sample, Dim<[usize; 1]>> {
        self.data.columns_mut()
    }

    #[rtsan::nonblocking]
    fn raw_buffer_mut(&mut self) -> &mut [Sample] {
        self.data.as_slice_memory_order_mut().unwrap()
    }

    #[rtsan::nonblocking]
    fn view_mut(&mut self) -> BlockViewMut<Sample> {
        BlockViewMut::from_array_view(self.data.view_mut(), self.sample_rate)
    }
}
