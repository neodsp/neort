#![cfg_attr(not(test), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

use core::{
    marker::PhantomData,
    ops::{Index, IndexMut},
};

#[cfg(feature = "alloc")]
use alloc::alloc::{alloc_zeroed, Layout};

pub use block_data::*;
use num_traits::Zero;
use rtsan_standalone::nonblocking;

pub mod block_data;
pub mod planar_copy_tools;

pub trait Num: Copy + Zero + PartialEq {}
impl<T: Copy + Zero + PartialEq> Num for T {}

#[cfg(feature = "alloc")]
pub type BlockHeap<T> = Block<Heap<T>>;
pub type BlockStack<T, const CAPACITY: usize> = Block<Stack<T, CAPACITY>>;
pub type BlockView<'a, T> = Block<View<'a, T>>;
pub type BlockViewMut<'a, T> = Block<ViewMut<'a, T>>;

pub struct Block<D: BlockDataConst> {
    data: D,
    num_channels: usize,
    num_frames: usize,
    channel_cap: usize,
    frame_cap: usize,
}

unsafe impl<D: BlockDataConst> Send for Block<D> {}
unsafe impl<D: BlockDataConst> Sync for Block<D> {}

impl<T: Num, const CAPACITY: usize> BlockStack<T, CAPACITY> {
    #[nonblocking]
    pub fn new(num_channels: usize, num_frames: usize) -> Self {
        assert!(num_channels * num_frames <= CAPACITY);
        Self {
            data: Stack {
                data: [T::zero(); CAPACITY],
            },
            num_channels,
            num_frames,
            channel_cap: num_channels,
            frame_cap: num_frames,
        }
    }
}

#[cfg(feature = "alloc")]
impl<T: Num> BlockHeap<T> {
    pub fn new(num_channels: usize, num_frames: usize) -> Self {
        let capacity = num_channels * num_frames;
        let layout = Layout::array::<T>(capacity).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) as *mut T };
        Self {
            data: Heap { ptr, capacity },
            num_channels,
            num_frames,
            channel_cap: num_channels,
            frame_cap: num_frames,
        }
    }
}

#[cfg(feature = "alloc")]
impl<T: Num> Default for BlockHeap<T> {
    fn default() -> Self {
        let capacity = 0;
        let layout = Layout::array::<T>(capacity).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) as *mut T };
        Self {
            data: Heap { ptr, capacity },
            num_frames: 0,
            num_channels: 0,
            channel_cap: 0,
            frame_cap: 0,
        }
    }
}

impl<'a, T: Num> BlockView<'a, T> {
    #[nonblocking]
    pub fn from_slice(slice: &'a [T], num_channels: usize, num_frames: usize) -> Self {
        Self::from_slice_limited(slice, num_channels, num_frames, num_channels, num_frames)
    }

    #[nonblocking]
    pub fn from_slice_limited(
        slice: &'a [T],
        num_channels: usize,
        num_frames: usize,
        channel_cap: usize,
        frame_cap: usize,
    ) -> Self {
        assert_eq!(slice.len(), channel_cap * frame_cap);
        Self::from_ptr_limited(
            slice.as_ptr(),
            num_channels,
            num_frames,
            channel_cap,
            frame_cap,
        )
    }

    #[nonblocking]
    pub fn from_ptr_limited(
        ptr: *const T,
        num_channels: usize,
        num_frames: usize,
        channel_cap: usize,
        frame_cap: usize,
    ) -> Self {
        assert!(num_channels <= channel_cap);
        assert!(num_frames <= frame_cap);
        Self {
            data: View {
                ptr,
                capacity: channel_cap * frame_cap,
                _phantom: PhantomData::<&'a T>,
            },
            num_channels,
            num_frames,
            channel_cap,
            frame_cap,
        }
    }
}

impl<'a, T: Num> BlockViewMut<'a, T> {
    #[nonblocking]
    pub fn from_slice(slice: &'a mut [T], num_channels: usize, num_frames: usize) -> Self {
        Self::from_slice_limited(slice, num_channels, num_frames, num_channels, num_frames)
    }

    #[nonblocking]
    pub fn from_slice_limited(
        slice: &'a mut [T],
        num_channels: usize,
        num_frames: usize,
        channel_cap: usize,
        frame_cap: usize,
    ) -> Self {
        assert_eq!(slice.len(), channel_cap * frame_cap);
        Self::from_ptr_limited(
            slice.as_mut_ptr(),
            num_channels,
            num_frames,
            channel_cap,
            frame_cap,
        )
    }

    #[nonblocking]
    pub fn from_ptr_limited(
        ptr: *mut T,
        num_channels: usize,
        num_frames: usize,
        channel_cap: usize,
        frame_cap: usize,
    ) -> Self {
        assert!(num_channels <= channel_cap);
        assert!(num_frames <= frame_cap);
        Self {
            data: ViewMut {
                ptr,
                capacity: channel_cap * frame_cap,
                _phantom: PhantomData::<&'a mut T>,
            },
            num_channels,
            num_frames,
            channel_cap,
            frame_cap,
        }
    }
}

impl<D: BlockDataConst + Sized> core::fmt::Debug for Block<D>
where
    D::Num: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!(
            "Block (channels: {}, frames: {})\n",
            self.num_channels, self.num_frames
        ))?;
        for (i, channel) in self.channels().enumerate() {
            f.write_fmt(format_args!("Channel {}: {:?}\n", i, channel))?;
        }
        Ok(())
    }
}

impl<D: BlockDataConst> Block<D> {
    #[nonblocking]
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    #[nonblocking]
    pub fn num_frames(&self) -> usize {
        self.num_frames
    }

    #[nonblocking]
    pub fn raw_data(&self) -> &[D::Num] {
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.data.capacity()) }
    }

    #[nonblocking]
    pub fn sample(&self, channel: usize, frame: usize) -> &D::Num {
        assert!(channel < self.num_channels);
        assert!(frame < self.num_frames);
        let index = channel * self.frame_cap + frame;
        unsafe { &*self.data.as_ptr().add(index) }
    }

    #[nonblocking]
    pub fn channel(&self, channel: usize) -> &[D::Num] {
        assert!(channel < self.num_channels);
        let start = channel * self.frame_cap;
        let len = self.num_frames;
        unsafe { core::slice::from_raw_parts(self.data.as_ptr().add(start), len) }
    }

    #[nonblocking]
    pub fn channels(&self) -> impl Iterator<Item = &[D::Num]> {
        (0..self.num_channels).map(|ch| {
            let start = ch * self.frame_cap;
            let len = self.num_frames;
            unsafe { core::slice::from_raw_parts(self.data.as_ptr().add(start), len) }
        })
    }

    #[nonblocking]
    pub fn view<'a>(&'a self) -> BlockView<'a, D::Num> {
        BlockView {
            data: View {
                ptr: self.data.as_ptr(),
                capacity: self.data.capacity(),
                _phantom: PhantomData::<&'a D::Num>,
            },
            num_channels: self.num_channels,
            num_frames: self.num_frames,
            channel_cap: self.channel_cap,
            frame_cap: self.frame_cap,
        }
    }
}

impl<D: BlockDataMut> Block<D> {
    #[nonblocking]
    pub fn set_num_channels_visible(&mut self, num_channels: usize) {
        assert!(num_channels <= self.channel_cap);
        self.num_channels = num_channels;
    }

    #[nonblocking]
    pub fn set_num_frames_visible(&mut self, num_frames: usize) {
        assert!(num_frames <= self.frame_cap);
        self.num_frames = num_frames;
    }

    #[nonblocking]
    pub fn raw_data_mut(&mut self) -> &mut [D::Num] {
        unsafe { core::slice::from_raw_parts_mut(self.data.as_mut_ptr(), self.data.capacity()) }
    }

    #[nonblocking]
    pub fn sample_mut(&mut self, channel: usize, frame: usize) -> &mut D::Num {
        assert!(channel < self.num_channels);
        assert!(frame < self.num_frames);
        let index = channel * self.frame_cap + frame;
        unsafe { &mut *self.data.as_mut_ptr().add(index) }
    }

    #[nonblocking]
    pub fn channel_mut(&mut self, channel: usize) -> &mut [D::Num] {
        assert!(channel < self.num_channels);
        let start = channel * self.frame_cap;
        let len = self.num_frames;
        unsafe { core::slice::from_raw_parts_mut(self.data.as_mut_ptr().add(start), len) }
    }

    #[nonblocking]
    pub fn channels_mut(&mut self) -> impl Iterator<Item = &mut [D::Num]> {
        (0..self.num_channels).map(|ch| {
            let start = ch * self.frame_cap;
            let len = self.num_frames;
            unsafe { core::slice::from_raw_parts_mut(self.data.as_mut_ptr().add(start), len) }
        })
    }

    #[nonblocking]
    pub fn copy_from_block<DO: BlockDataConst<Num = D::Num>>(&mut self, other: &Block<DO>) {
        self.set_num_channels_visible(other.num_channels);
        self.set_num_frames_visible(other.num_frames);
        for (this_ch, other_ch) in self.channels_mut().zip(other.channels()) {
            this_ch.copy_from_slice(other_ch);
        }
    }

    #[nonblocking]
    pub fn clear(&mut self) {
        self.raw_data_mut().fill(D::Num::zero());
    }

    #[nonblocking]
    pub fn view_mut<'a>(&'a mut self) -> BlockViewMut<'a, D::Num> {
        BlockViewMut {
            data: ViewMut {
                ptr: self.data.as_mut_ptr(),
                capacity: self.data.capacity(),
                _phantom: PhantomData::<&'a mut D::Num>,
            },
            num_channels: self.num_channels,
            num_frames: self.num_frames,
            channel_cap: self.channel_cap,
            frame_cap: self.frame_cap,
        }
    }
}

impl<D: BlockDataConst> Index<[usize; 2]> for Block<D> {
    type Output = D::Num;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        self.sample(index[0], index[1])
    }
}

impl<D: BlockDataMut> IndexMut<[usize; 2]> for Block<D> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        self.sample_mut(index[0], index[1])
    }
}

impl<D: BlockDataConst> Index<usize> for Block<D> {
    type Output = [D::Num];

    fn index(&self, index: usize) -> &Self::Output {
        self.channel(index)
    }
}

impl<D: BlockDataMut> IndexMut<usize> for Block<D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.channel_mut(index)
    }
}

impl<D1: BlockDataConst, D2: BlockDataConst<Num = D1::Num>> PartialEq<Block<D2>> for Block<D1> {
    fn eq(&self, other: &Block<D2>) -> bool {
        if self.num_channels != other.num_channels || self.num_frames != other.num_frames {
            return false;
        }
        self.channels()
            .zip(other.channels())
            .all(|(this_ch, other_ch)| this_ch == other_ch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let mut block = BlockStack::<i32, 10>::new(2, 5);
        assert_eq!(block.raw_data(), &[0; 10]);
        assert_eq!(block.raw_data_mut(), &[0; 10]);

        let mut block = BlockHeap::<i32>::new(2, 5);
        assert_eq!(block.raw_data(), &[0; 10]);
        assert_eq!(block.raw_data_mut(), &[0; 10]);

        let mut data = [0; 10];
        let block = BlockView::<i32>::from_slice(&data, 2, 5);
        assert_eq!(block.raw_data(), &[0; 10]);

        let mut block = BlockViewMut::<i32>::from_slice(&mut data, 2, 5);
        assert_eq!(block.raw_data(), &[0; 10]);
        assert_eq!(block.raw_data_mut(), &[0; 10]);
    }

    #[test]
    fn test_sample() {
        let mut block = BlockHeap::<i32>::new(2, 3);
        block[[0, 1]] = 24;
        block[[1, 2]] = 42;
        assert_eq!(block.raw_data(), &[0, 24, 0, 0, 0, 42]);
        assert_eq!(block[[0, 1]], 24);
        assert_eq!(block[[1, 2]], 42);

        let mut block = BlockHeap::<i32>::new(2, 3);
        *block.sample_mut(0, 1) = 24;
        *block.sample_mut(1, 2) = 42;
        assert_eq!(block.raw_data(), &[0, 24, 0, 0, 0, 42]);
        assert_eq!(*block.sample(0, 1), 24);
        assert_eq!(*block.sample(1, 2), 42);
    }

    #[test]
    fn test_channels() {
        let mut block = BlockHeap::<i32>::new(2, 3);
        block[0].fill(24);
        assert_eq!(block.raw_data(), &[24, 24, 24, 0, 0, 0]);
        block[1].fill(42);
        assert_eq!(block.raw_data(), &[24, 24, 24, 42, 42, 42]);
        assert_eq!(block[0], [24, 24, 24]);
        assert_eq!(block[1], [42, 42, 42]);

        let mut block = BlockHeap::<i32>::new(2, 3);
        block.channel_mut(0).fill(24);
        assert_eq!(block.raw_data(), &[24, 24, 24, 0, 0, 0]);
        block.channel_mut(1).fill(42);
        assert_eq!(block.raw_data(), &[24, 24, 24, 42, 42, 42]);
        assert_eq!(block.channel(0), [24, 24, 24]);
        assert_eq!(block.channel(1), [42, 42, 42]);
    }

    #[test]
    fn test_channel_iters() {
        let mut block = BlockHeap::<usize>::new(2, 3);
        for (ch_idx, channel) in block.channels_mut().enumerate() {
            channel.fill(ch_idx + 1);
        }
        assert_eq!(block.raw_data(), &[1, 1, 1, 2, 2, 2]);

        for (ch_idx, channel) in block.channels().enumerate() {
            assert_eq!(channel, &[ch_idx + 1; 3]);
        }
    }

    #[test]
    fn test_copy_from() {
        let mut block = BlockHeap::<i32>::new(2, 3);

        let block2 = BlockView::<i32>::from_slice(&[0, 1, 2, 3, 4, 5], 2, 3);

        block.copy_from_block(&block2);

        assert_eq!(block.raw_data(), block2.raw_data());
    }

    #[test]
    fn test_view() {
        let mut block = BlockHeap::<usize>::new(2, 3);
        {
            let mut view_mut = block.view_mut();
            for (ch_idx, channel) in view_mut.channels_mut().enumerate() {
                channel.fill(ch_idx + 1);
            }

            assert_eq!(view_mut[0], [1, 1, 1]);
            assert_eq!(view_mut[1], [2, 2, 2]);
        }

        let view = block.view();

        assert_eq!(view[0], [1, 1, 1]);
        assert_eq!(view[1], [2, 2, 2]);
    }

    #[test]
    fn test_equality() {
        let mut block1 = BlockHeap::<i32>::new(2, 3);
        let block2 = BlockView::<i32>::from_slice(&[0, 1, 2, 3, 4, 5], 2, 3);
        block1.copy_from_block(&block2);
        assert_eq!(block1, block2);

        block1[[1, 1]] = 0;
        assert_ne!(block1, block2);
    }
}
