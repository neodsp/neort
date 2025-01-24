use rtsan_standalone::nonblocking;

use crate::{Block, BlockDataConst, BlockDataMut};

// Planar Copy Tools
impl<D: BlockDataConst> Block<D> {
    #[nonblocking]
    #[allow(clippy::missing_safety_doc)]
    pub fn copy_into_planar_data<V: AsMut<[D::Num]>>(&self, data: &mut [V]) {
        let num_channels = data.len();
        let num_frames = data.first_mut().map_or(0, |channel| channel.as_mut().len());
        assert_eq!(self.num_channels, num_channels);
        assert_eq!(self.num_frames, num_frames);
        for (this_ch, channel) in self.channels().zip(data) {
            channel.as_mut().copy_from_slice(this_ch);
        }
    }

    #[nonblocking]
    #[allow(clippy::missing_safety_doc)]
    pub fn copy_into_planar_data_limited<V: AsMut<[D::Num]>>(
        &self,
        data: &mut [V],
        num_channels: usize,
        num_frames: usize,
    ) {
        assert!(num_channels <= data.len());
        assert_eq!(self.num_channels, num_channels);
        assert_eq!(self.num_frames, num_frames);
        for (this_ch, channel) in self.channels().zip(data) {
            channel.as_mut()[..num_frames].copy_from_slice(this_ch);
        }
    }

    #[nonblocking]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn copy_into_planar_ptr(
        &mut self,
        ptr: *const *mut D::Num,
        num_channels: usize,
        num_frames: usize,
    ) {
        assert_eq!(self.num_channels, num_channels);
        assert_eq!(self.num_frames, num_frames);
        let channel_ptrs = core::slice::from_raw_parts(ptr, num_channels);
        for (this_ch, ch_ptr) in self.channels().zip(channel_ptrs) {
            core::slice::from_raw_parts_mut(*ch_ptr, num_frames).copy_from_slice(this_ch);
        }
    }
}

// Planar Copy Tools
impl<D: BlockDataMut> Block<D> {
    #[nonblocking]
    #[allow(clippy::missing_safety_doc)]
    pub fn copy_from_planar_data<V: AsRef<[D::Num]>>(&mut self, data: &[V]) {
        let num_channels = data.len();
        let num_frames = data.first().map_or(0, |channel| channel.as_ref().len());
        self.set_num_channels_visible(num_channels);
        self.set_num_frames_visible(num_frames);
        for (this_ch, channel) in self.channels_mut().zip(data) {
            this_ch.copy_from_slice(channel.as_ref());
        }
    }

    #[nonblocking]
    #[allow(clippy::missing_safety_doc)]
    pub fn copy_from_planar_data_limited<V: AsRef<[D::Num]>>(
        &mut self,
        data: &[V],
        num_channels: usize,
        num_frames: usize,
    ) {
        assert!(num_channels <= data.len());
        self.set_num_channels_visible(num_channels);
        self.set_num_frames_visible(num_frames);
        for (this_ch, channel) in self.channels_mut().zip(data) {
            this_ch.copy_from_slice(&channel.as_ref()[..num_frames]);
        }
    }

    #[nonblocking]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn copy_from_planar_ptr(
        &mut self,
        ptr: *const *const D::Num,
        num_channels: usize,
        num_frames: usize,
    ) {
        self.set_num_channels_visible(num_channels);
        self.set_num_frames_visible(num_frames);
        let channel_ptrs = core::slice::from_raw_parts(ptr, num_channels);
        for (this_ch, ch_ptr) in self.channels_mut().zip(channel_ptrs) {
            this_ch.copy_from_slice(core::slice::from_raw_parts(*ch_ptr, num_frames));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::BlockHeap;

    #[test]
    fn test_planar() {
        let planar = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]];
        let mut block = BlockHeap::<i32>::new(3, 6);
        block.copy_from_planar_data(&planar);
        assert_eq!(block[0], [0, 1, 2, 3, 4]);
        assert_eq!(block[1], [5, 6, 7, 8, 9]);

        let mut planar = [[0; 5], [0; 5]];
        block.copy_into_planar_data(&mut planar);
        assert_eq!(planar[0], [0, 1, 2, 3, 4]);
        assert_eq!(planar[1], [5, 6, 7, 8, 9]);

        let planar = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]];
        let mut block = BlockHeap::<i32>::new(3, 6);
        block.copy_from_planar_data_limited(&planar, 2, 4);
        assert_eq!(block[0], [0, 1, 2, 3]);
        assert_eq!(block[1], [5, 6, 7, 8]);

        let mut planar = [[0; 5], [0; 5], [0; 5]];
        block.copy_into_planar_data_limited(&mut planar, 2, 4);
        assert_eq!(planar[0], [0, 1, 2, 3, 0]);
        assert_eq!(planar[1], [5, 6, 7, 8, 0]);
        assert_eq!(planar[2], [0, 0, 0, 0, 0]);
    }
}
