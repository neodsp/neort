use ndarray::{s, Array2, ArrayView1};
use num::Float;
use rtsan::{blocking, nonblocking};

use super::{block_view::BlockView, BlockViewMut};

#[derive(Debug, Clone, PartialEq)]
pub struct Block<F: Float> {
    data: Array2<F>,
    num_frames_visible: usize,
}

impl<F: Float> Default for Block<F> {
    #[blocking]
    fn default() -> Self {
        Self {
            data: Array2::zeros((0, 0)),
            num_frames_visible: 0,
        }
    }
}

impl<F: Float> Block<F> {
    #[blocking]
    pub fn new(num_channels: u16, num_frames: usize) -> Self {
        Self {
            data: Array2::zeros((num_channels as usize, num_frames)),
            num_frames_visible: num_frames,
        }
    }

    #[blocking]
    pub fn from_array(array: Array2<F>) -> Self {
        Self {
            num_frames_visible: array.ncols(),
            data: array,
        }
    }

    /// This returns the maximum amount of samples that can be stored
    /// The owned block can operate on less samples, than available
    /// and [ `num_frames` ] just returns the number of frames that
    /// should be accessed.
    #[nonblocking]
    pub fn num_frames_max(&self) -> usize {
        self.data.ncols()
    }

    // TODO: Test real-time-save "resizing"
    #[nonblocking]
    pub fn set_num_frames_accesible(&mut self, num_frames: usize) {
        self.num_frames_visible = num_frames;
    }

    // TODO: Test
    /// # Safety
    /// The function is safe to call as long as the ptr has `num_channels` elements
    /// and each of those pointers is pointing to `num_frames` values.
    /// It is undefined behaviour if this is not the case!
    #[nonblocking]
    pub unsafe fn copy_from_ptr(
        &mut self,
        ptr: *const *mut F,
        num_channels: u16,
        num_frames: usize,
    ) {
        assert_eq!(num_channels, self.num_channels());
        // writing can resize the number of frames
        assert!(num_frames <= self.num_frames_max());
        self.num_frames_visible = num_frames;

        let raw_slices = std::slice::from_raw_parts(ptr, num_channels as usize);

        for (channel_idx, &channel_ptr) in raw_slices.iter().enumerate() {
            let channel_data = std::slice::from_raw_parts(channel_ptr, num_frames);
            let mut channel_view = self.data.slice_mut(ndarray::s![channel_idx, ..num_frames]);
            channel_view.assign(&ArrayView1::from_shape(num_frames, channel_data).unwrap());
        }
    }

    // TODO: Test
    /// # Safety
    /// The function is safe to call as long as the ptr has `num_channels` elements
    /// and each of those pointers is pointing to `num_frames` values.
    /// It is undefined behaviour if this is not the case!
    #[nonblocking]
    pub unsafe fn copy_into_ptr(&self, ptr: *const *mut F, num_channels: u16, num_frames: usize) {
        assert_eq!(num_channels, self.num_channels());
        // reading should be done with the the same number of frames that were stored last
        assert_eq!(num_frames, self.num_frames());

        let raw_slices = std::slice::from_raw_parts(ptr, num_channels as usize);

        for (channel_idx, &channel_ptr) in raw_slices.iter().enumerate() {
            let channel_data = self.data.slice(ndarray::s![channel_idx, ..num_frames]);
            let output_slice = std::slice::from_raw_parts_mut(channel_ptr, num_frames);
            output_slice.copy_from_slice(channel_data.as_slice().unwrap());
        }
    }

    // TODO: Test
    #[nonblocking]
    pub fn view(&self) -> BlockView<F> {
        BlockView::from_array_view(self.data.slice(s![.., ..self.num_frames_visible]))
    }

    // TODO: Test
    #[nonblocking]
    pub fn view_mut(&mut self) -> BlockViewMut<F> {
        BlockViewMut::from_array_view(self.data.slice_mut(s![.., ..self.num_frames_visible]))
    }

    // TODO: Test
    #[nonblocking]
    pub fn num_channels(&self) -> u16 {
        self.data.nrows() as u16
    }

    // TODO: Test
    #[nonblocking]
    pub fn num_frames(&self) -> usize {
        self.num_frames_visible
    }

    // TODO: Test
    #[nonblocking]
    pub fn num_frames_allocated(&self) -> usize {
        self.data.ncols()
    }

    // TODO: Test
    /// Raw buffers are only for special purposes, reading should be done by taking a `BlocKView`.
    /// The reason for this is that the view will only give you the data that is meant to read from
    /// and not the whole allocated storage.
    #[nonblocking]
    pub fn raw_buffer(&self) -> &[F] {
        self.data.as_slice_memory_order().unwrap()
    }

    // TODO: Test
    /// Raw buffers are only for special purposes, writing should be done by taking a `BlocKViewMut`.
    /// The reason for this is that the view will only give you the data that is meant to write to
    /// and not the whole allocated storage.
    #[nonblocking]
    pub fn raw_buffer_mut(&mut self) -> &mut [F] {
        self.data.as_slice_memory_order_mut().unwrap()
    }

    // TODO: Test
    #[nonblocking]
    pub fn clear(&mut self) {
        self.data.fill(F::zero());
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, aview2};

    use super::*;

    #[test]
    fn block() {
        let block = Block::<f32>::new(2, 3);
        assert_eq!(
            block.view(),
            BlockView::from_array_view(aview2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
        );

        let block = Block::<f32>::from_array(array![[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]);

        // Block Read
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
    }
}
