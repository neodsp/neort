use ndarray::{s, Array2, ArrayView1};
use rtsan::{blocking, nonblocking};

use crate::Sample;

use super::{BlockRead, BlockView, BlockViewMut};

#[derive(Debug, Clone, PartialEq)]
pub struct Block<S: Sample> {
    data: Array2<S>,
    num_frames_visible: usize,
}

impl<S: Sample> Default for Block<S> {
    #[blocking]
    fn default() -> Self {
        Self {
            data: Array2::zeros((0, 0)),
            num_frames_visible: 0,
        }
    }
}

impl<S: Sample> Block<S> {
    #[blocking]
    pub fn new(num_channels: u16, num_frames_max: usize) -> Self {
        Self {
            data: Array2::zeros((num_channels as usize, num_frames_max)),
            num_frames_visible: num_frames_max,
        }
    }

    #[blocking]
    pub fn from_array(array: Array2<S>) -> Self {
        Self {
            num_frames_visible: array.ncols(),
            data: array.as_standard_layout().to_owned(),
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

    #[nonblocking]
    pub fn set_num_frames(&mut self, num_frames: usize) {
        self.num_frames_visible = num_frames;
    }

    /// # Safety
    /// The function is safe to call as long as the ptr has `num_channels` elements
    /// and each of those pointers is pointing to `num_frames` values.
    /// It is undefined behaviour if this is not the case!
    #[nonblocking]
    pub unsafe fn copy_from_ptr(
        &mut self,
        ptr: *const *mut S,
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

    /// # Safety
    /// The function is safe to call as long as the ptr has `num_channels` elements
    /// and each of those pointers is pointing to `num_frames` values.
    /// It is undefined behaviour if this is not the case!
    #[nonblocking]
    pub unsafe fn copy_into_ptr(&self, ptr: *const *mut S, num_channels: u16, num_frames: usize) {
        assert_eq!(num_channels, self.num_channels());
        // reading should be done with the the same number of frames that were stored last
        assert_eq!(num_frames, self.num_frames());

        let raw_slices = std::slice::from_raw_parts(ptr, num_channels as usize);

        for (channel_idx, &channel_ptr) in raw_slices.iter().enumerate() {
            let channel_data = self.data.slice(ndarray::s![channel_idx, ..num_frames]);
            let output_slice = std::slice::from_raw_parts_mut(channel_ptr, num_frames);
            for (out, inp) in output_slice.iter_mut().zip(channel_data.iter()) {
                *out = *inp;
            }
        }
    }

    /// TODO: Test
    #[nonblocking]
    pub fn copy_from_slices(&mut self, slice: &mut [&mut [S]]) {
        assert_eq!(slice.len(), self.num_channels() as usize);
        // writing can resize the number of frames
        let num_frames = slice[0].len();
        assert!(
            num_frames <= self.num_frames_max(),
            "Not enough memory allocated in block"
        );
        self.num_frames_visible = num_frames;

        for (channel_idx, channel_data) in slice.iter().enumerate() {
            assert_eq!(
                channel_data.len(),
                num_frames,
                "All frames must be equally long"
            );

            let mut channel_view = self.data.slice_mut(ndarray::s![channel_idx, ..num_frames]);
            channel_view.assign(&ArrayView1::from_shape(num_frames, channel_data).unwrap());
        }
    }

    /// TODO: Test
    #[nonblocking]
    pub fn copy_into_slices(&self, slice: &mut [&mut [S]]) {
        assert_eq!(slice.len(), self.num_channels() as usize);
        // writing can resize the number of frames
        let num_frames = slice[0].len();
        assert_eq!(num_frames, self.num_frames_max());

        for (channel_idx, output_slice) in slice.iter_mut().enumerate() {
            assert_eq!(
                output_slice.len(),
                num_frames,
                "All frames must be equally long"
            );

            let channel_view = self.data.slice(ndarray::s![channel_idx, ..num_frames]);
            for (out, inp) in output_slice.iter_mut().zip(channel_view.iter()) {
                *out = *inp;
            }
        }
    }
    #[nonblocking]
    pub fn num_frames_allocated(&self) -> usize {
        self.data.ncols()
    }

    /// Raw buffers are only for special purposes, reading should be done by taking a `BlocKView`.
    /// The reason for this is that the view will only give you the data that is meant to read from
    /// and not the whole allocated storage.
    #[nonblocking]
    pub fn raw_buffer(&self) -> &[S] {
        self.data.as_slice_memory_order().unwrap()
    }

    /// Raw buffers are only for special purposes, writing should be done by taking a `BlocKViewMut`.
    /// The reason for this is that the view will only give you the data that is meant to write to
    /// and not the whole allocated storage.
    #[nonblocking]
    pub fn raw_buffer_mut(&mut self) -> &mut [S] {
        self.data.as_slice_memory_order_mut().unwrap()
    }

    #[nonblocking]
    pub fn view(&self) -> BlockView<S> {
        BlockView::from_array_view(self.data.slice(s![.., ..self.num_frames_visible]))
    }

    #[nonblocking]
    pub fn view_mut(&mut self) -> BlockViewMut<S> {
        BlockViewMut::from_array_view(self.data.slice_mut(s![.., ..self.num_frames_visible]))
    }

    #[nonblocking]
    pub fn sample_mut(&mut self, ch: u16, frame: usize) -> &mut S {
        assert!(frame < self.num_frames_visible);
        &mut self.data[[ch as usize, frame]]
    }

    #[nonblocking]
    pub fn channel_mut(&mut self, index: u16) -> ndarray::ArrayViewMut1<S> {
        self.data
            .slice_mut(s![index as usize, ..self.num_frames_visible])
    }

    #[nonblocking]
    pub fn clear(&mut self) {
        self.data.fill(S::zero());
    }
}

impl<S: Sample> BlockRead<S> for Block<S> {
    fn num_channels(&self) -> u16 {
        self.data.nrows() as u16
    }

    fn num_frames(&self) -> usize {
        self.num_frames_visible
    }

    fn sample(&self, ch: u16, frame: usize) -> S {
        assert!(frame < self.num_frames_visible);
        self.data[[ch as usize, frame]]
    }

    fn channel(&self, index: u16) -> ArrayView1<S> {
        self.data
            .slice(s![index as usize, ..self.num_frames_visible])
    }

    fn channels(&self) -> impl Iterator<Item = ndarray::ArrayView1<S>> {
        (0..self.num_channels())
            .into_iter()
            .map(|ch| self.channel(ch))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn block() {
        // Empty
        let mut block = Block::<f32>::new(2, 3);
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_frames_max(), 3);

        // Change num frames
        block.set_num_frames(2);
        assert_eq!(block.num_frames(), 2);
        assert_eq!(block.num_frames_allocated(), 3);

        // From Array
        let array = array![[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]];
        let block = Block::<f32>::from_array(array.clone());
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        // assert_eq!(block.view(), BlockView::from_array_view(array.view()));

        // Copy from pointer
        let mut block = Block::<f32>::new(2, 5);
        let data: [[f32; 3]; 2] = [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]];
        let ptrs: Vec<*mut f32> = data.iter().map(|d| d.as_ptr() as *mut f32).collect();

        unsafe {
            block.copy_from_ptr(ptrs.as_ptr(), 2, 3);
        }

        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_frames_allocated(), 5);

        // let mut expected = array![[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]];
        // assert_eq!(block.view(), BlockView::from_array_view(expected.view()));

        // assert_eq!(
        //     block.view_mut(),
        //     BlockViewMut::from_array_view(expected.view_mut())
        // );
        assert_eq!(
            block.raw_buffer_mut(),
            &[0.1, 0.2, 0.3, 0.0, 0.0, 1.1, 1.2, 1.3, 0.0, 0.0]
        );

        // Interleaved
    }
}
