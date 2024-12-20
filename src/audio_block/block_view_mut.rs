use ndarray::{ArrayView1, ArrayViewMut1, ArrayViewMut2, ShapeBuilder};
use rtsan::nonblocking;

use crate::Sample;

use super::{BlockRead, BlockWrite, BufferLayout};

#[derive(Debug, PartialEq)]
pub struct BlockViewMut<'a, S: Sample> {
    data: ArrayViewMut2<'a, S>,
}

impl<S: Sample> Default for BlockViewMut<'_, S> {
    #[nonblocking]
    fn default() -> Self {
        Self {
            data: ArrayViewMut2::from_shape((0, 0), &mut []).unwrap(),
        }
    }
}

impl<'a, S: Sample> BlockViewMut<'a, S> {
    #[nonblocking]
    pub fn from_buffer(
        buffer: &'a mut [S],
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
    pub fn from_array_view(view: ArrayViewMut2<'a, S>) -> Self {
        Self { data: view }
    }

    // TODO: Test
    /// # Safety
    /// The buffer the pointer points to must be at least `num_channels` * `num_frames`
    /// elements long. Otherwise this is undefined behavior.
    #[nonblocking]
    #[inline(always)]
    pub unsafe fn from_ptr(ptr: *mut S, num_channels: u16, num_frames: usize) -> Self {
        Self {
            data: ArrayViewMut2::from_shape_ptr((num_channels as usize, num_frames), ptr),
        }
    }
}

impl<S: Sample> BlockRead<S> for BlockViewMut<'_, S> {
    fn num_channels(&self) -> u16 {
        self.data.nrows() as u16
    }

    fn num_frames(&self) -> usize {
        self.data.ncols()
    }

    fn sample(&self, ch: u16, frame: usize) -> S {
        self.data[[ch as usize, frame]]
    }

    fn channel(&self, index: u16) -> ArrayView1<S> {
        self.data.row(index as usize)
    }

    fn channels(&self) -> impl Iterator<Item = ndarray::ArrayView1<S>> {
        self.data.rows().into_iter()
    }
}

impl<S: Sample> BlockWrite<S> for BlockViewMut<'_, S> {
    fn sample_mut(&mut self, ch: u16, frame: usize) -> &mut S {
        &mut self.data[[ch as usize, frame]]
    }

    fn channel_mut(&mut self, index: u16) -> ArrayViewMut1<S> {
        self.data.row_mut(index as usize)
    }

    fn channels_mut(&mut self) -> impl Iterator<Item = ndarray::ArrayViewMut1<S>> {
        self.data.rows_mut().into_iter()
    }
}
