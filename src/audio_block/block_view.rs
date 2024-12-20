use ndarray::{ArrayView1, ArrayView2, ShapeBuilder};
use rtsan::nonblocking;

use crate::Sample;

use super::{BlockRead, BufferLayout};

#[derive(Debug, Clone, PartialEq)]
pub struct BlockView<'a, S: Sample> {
    data: ArrayView2<'a, S>,
}

impl<S: Sample> Default for BlockView<'_, S> {
    #[nonblocking]
    fn default() -> Self {
        Self {
            data: ArrayView2::from_shape((0, 0), &[]).unwrap(),
        }
    }
}

impl<'a, S: Sample> BlockView<'a, S> {
    #[nonblocking]
    #[inline(always)]
    pub fn from_buffer(
        buffer: &'a [S],
        num_channels: u16,
        num_frames: usize,
        layout: BufferLayout,
    ) -> Self {
        let data = match layout {
            BufferLayout::Sequential => {
                ArrayView2::from_shape((num_channels as usize, num_frames), buffer).unwrap()
            }
            BufferLayout::Interleaved => {
                ArrayView2::from_shape((num_channels as usize, num_frames).f(), buffer).unwrap()
            }
        };
        Self { data }
    }

    #[nonblocking]
    #[inline(always)]
    pub fn from_array_view(view: ArrayView2<'a, S>) -> Self {
        Self { data: view }
    }

    // TODO: Test
    /// # Safety
    /// The buffer the pointer points to must be at least `num_channels` * `num_frames`
    /// elements long. Otherwise this is undefined behavior.
    #[nonblocking]
    #[inline(always)]
    pub unsafe fn from_ptr(ptr: *const S, num_channels: u16, num_frames: usize) -> Self {
        Self {
            data: ArrayView2::from_shape_ptr((num_channels as usize, num_frames), ptr),
        }
    }
}

impl<S: Sample> BlockRead<S> for BlockView<'_, S> {
    #[nonblocking]
    fn num_channels(&self) -> u16 {
        self.data.nrows() as u16
    }

    #[nonblocking]
    fn num_frames(&self) -> usize {
        self.data.ncols()
    }

    #[nonblocking]
    fn sample(&self, ch: u16, frame: usize) -> S {
        self.data[[ch as usize, frame]]
    }

    #[nonblocking]
    fn channel(&self, index: u16) -> ArrayView1<S> {
        self.data.row(index as usize)
    }

    #[nonblocking]
    fn channels(&self) -> impl Iterator<Item = ndarray::ArrayView1<S>> {
        self.data.rows().into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, array};

    #[test]
    fn test_default() {
        let block: BlockView<f32> = BlockView::default();
        assert_eq!(block.num_channels(), 0);
        assert_eq!(block.num_frames(), 0);
    }

    #[test]
    fn test_from_buffer_sequential() {
        let buffer = [1.0f32, 2.0, 3.0, 4.0];
        let block = BlockView::from_buffer(&buffer, 2, 2, BufferLayout::Sequential);
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 2);
        assert_eq!(block.sample(0, 0), 1.0);
        assert_eq!(block.sample(1, 1), 4.0);
    }

    #[test]
    fn test_from_buffer_interleaved() {
        let buffer = [1.0f32, 2.0, 3.0, 4.0];
        let block = BlockView::from_buffer(&buffer, 2, 2, BufferLayout::Interleaved);
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 2);
        assert_eq!(block.sample(0, 0), 1.0);
        assert_eq!(block.sample(1, 0), 2.0);
        assert_eq!(block.sample(0, 1), 3.0);
        assert_eq!(block.sample(1, 1), 4.0);
    }

    #[test]
    fn test_from_array_view() {
        let arr = arr2(&[[1.0f32, 2.0], [3.0, 4.0]]);
        let block = BlockView::from_array_view(arr.view());
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 2);
        assert_eq!(block.sample(0, 1), 2.0);
        assert_eq!(block.sample(1, 0), 3.0);
    }

    #[test]
    fn test_from_ptr() {
        let buffer = [1.0f32, 2.0, 3.0, 4.0];
        let ptr = buffer.as_ptr();
        let block = unsafe { BlockView::from_ptr(ptr, 2, 2) };
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 2);
        assert_eq!(block.sample(0, 0), 1.0);
        assert_eq!(block.sample(1, 1), 4.0);
    }

    #[test]
    fn test_num_channels() {
        let arr = arr2(&[[1.0f32, 2.0], [3.0, 4.0]]);
        let block = BlockView::from_array_view(arr.view());
        assert_eq!(block.num_channels(), 2);
    }

    #[test]
    fn test_num_frames() {
        let arr = arr2(&[[1.0f32, 2.0], [3.0, 4.0]]);
        let block = BlockView::from_array_view(arr.view());
        assert_eq!(block.num_frames(), 2);
    }

    #[test]
    fn test_sample() {
        let arr = arr2(&[[10.0f32, 20.0], [30.0, 40.0]]);
        let block = BlockView::from_array_view(arr.view());
        assert_eq!(block.sample(0, 1), 20.0);
        assert_eq!(block.sample(1, 0), 30.0);
    }

    #[test]
    fn test_channel() {
        let arr = arr2(&[[10.0f32, 20.0], [30.0, 40.0]]);
        let block = BlockView::from_array_view(arr.view());
        let c0 = block.channel(0);
        let c1 = block.channel(1);
        assert_eq!(c0, array![10.0, 20.0]);
        assert_eq!(c1, array![30.0, 40.0]);
    }

    #[test]
    fn test_channels() {
        let arr = arr2(&[[10.0f32, 20.0], [30.0, 40.0]]);
        let block = BlockView::from_array_view(arr.view());
        let chans: Vec<_> = block.channels().collect();
        assert_eq!(chans.len(), 2);
        assert_eq!(chans[0], array![10.0, 20.0]);
        assert_eq!(chans[1], array![30.0, 40.0]);
    }
}
