use rtsan::nonblocking;

use crate::{Block, BlockDataConst, BlockDataMut};

// Planar Copy Tools
impl<D: BlockDataConst> Block<D> {
    #[nonblocking]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn copy_into_planar_data<V: AsMut<[D::Num]>>(&self, data: &mut [V]) {
        let num_channels = data.len() as u16;
        let num_frames = data.first_mut().map_or(0, |channel| channel.as_mut().len()) as u32;
        assert_eq!(self.num_channels, num_channels);
        assert_eq!(self.num_frames, num_frames);
        for (this_ch, channel) in self.channel_iter().zip(data) {
            channel.as_mut().copy_from_slice(this_ch);
        }
    }

    #[nonblocking]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn copy_into_planar_data_limited<V: AsMut<[D::Num]>>(
        &self,
        data: &mut [V],
        num_channels: u16,
        num_frames: u32,
    ) {
        assert!(num_channels <= data.len() as u16);
        assert_eq!(self.num_channels, num_channels);
        assert_eq!(self.num_frames, num_frames);
        for (this_ch, channel) in self.channel_iter().zip(data) {
            channel.as_mut()[..num_frames as usize].copy_from_slice(this_ch);
        }
    }

    #[nonblocking]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn copy_into_planar_ptr(
        &mut self,
        ptr: *const *mut D::Num,
        num_channels: u16,
        num_frames: u32,
    ) {
        assert_eq!(self.num_channels, num_channels);
        assert_eq!(self.num_frames, num_frames);
        let channel_ptrs = core::slice::from_raw_parts(ptr, num_channels as usize);
        for (this_ch, ch_ptr) in self.channel_iter().zip(channel_ptrs) {
            core::slice::from_raw_parts_mut(*ch_ptr, num_frames as usize).copy_from_slice(this_ch);
        }
    }
}

// Planar Copy Tools
impl<D: BlockDataMut> Block<D> {
    #[nonblocking]
    #[allow(clippy::missing_safety_doc)]
    pub fn copy_from_planar_data<V: AsRef<[D::Num]>>(&mut self, data: &[V]) {
        let num_channels = data.len() as u16;
        let num_frames = data.first().map_or(0, |channel| channel.as_ref().len()) as u32;
        self.set_num_channels_visible(num_channels);
        self.set_num_frames_visible(num_frames);
        for (this_ch, channel) in self.channel_iter_mut().zip(data) {
            this_ch.copy_from_slice(channel.as_ref());
        }
    }

    #[nonblocking]
    #[allow(clippy::missing_safety_doc)]
    pub fn copy_from_planar_data_limited<V: AsRef<[D::Num]>>(
        &mut self,
        data: &[V],
        num_channels: u16,
        num_frames: u32,
    ) {
        assert!(num_channels as usize <= data.len());
        self.set_num_channels_visible(num_channels);
        self.set_num_frames_visible(num_frames);
        for (this_ch, channel) in self.channel_iter_mut().zip(data) {
            this_ch.copy_from_slice(&channel.as_ref()[..num_frames as usize]);
        }
    }

    #[nonblocking]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn copy_from_planar_ptr(
        &mut self,
        ptr: *const *const D::Num,
        num_channels: u16,
        num_frames: u32,
    ) {
        self.set_num_channels_visible(num_channels);
        self.set_num_frames_visible(num_frames);
        let channel_ptrs = core::slice::from_raw_parts(ptr, num_channels as usize);
        for (this_ch, ch_ptr) in self.channel_iter_mut().zip(channel_ptrs) {
            this_ch.copy_from_slice(core::slice::from_raw_parts(*ch_ptr, num_frames as usize));
        }
    }
}
