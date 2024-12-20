use ndarray::{ArrayView1, ArrayViewMut1};
use rtsan::nonblocking;

use crate::Sample;

use super::{BlockRead, BlockWrite};

pub struct BlockViewStackedMut<'a, S: Sample> {
    data: &'a mut [*mut S],
    num_frames: usize,
}

impl<'a, F: Sample> BlockViewStackedMut<'a, F> {
    #[nonblocking]
    pub fn from_ptr(data: *const *mut F, channels: usize, samples: usize) -> Self {
        let data = unsafe { std::slice::from_raw_parts_mut(data as *mut *mut F, channels) };
        Self {
            data,
            num_frames: samples,
        }
    }

    #[nonblocking]
    pub fn from_slices<'b>(channels: &'a mut [&'b mut [F]]) -> Self {
        let num_channels = channels.len();
        let num_frames = channels[0].len();
        for ch in channels.iter() {
            assert_eq!(ch.len(), num_frames);
        }

        let ptr = channels.as_mut_ptr();
        let pointers = unsafe { std::slice::from_raw_parts_mut(ptr as *mut *mut F, num_channels) };

        for (ch, p) in channels.iter_mut().zip(pointers.iter_mut()) {
            *p = ch.as_mut_ptr();
        }

        Self::from_ptr(pointers.as_ptr(), num_channels, num_frames)
    }
}

impl<'a, S: Sample> BlockRead<S> for BlockViewStackedMut<'a, S> {
    #[nonblocking]
    fn num_channels(&self) -> u16 {
        self.data.len() as u16
    }

    #[nonblocking]
    fn num_frames(&self) -> usize {
        self.num_frames
    }

    #[nonblocking]
    fn sample(&self, ch: u16, frame: usize) -> S {
        unsafe { *self.data[ch as usize].add(frame) }
    }

    #[nonblocking]
    fn channel(&self, index: u16) -> ArrayView1<S> {
        unsafe { ArrayView1::from_shape_ptr(self.num_frames, self.data[index as usize]) }
    }

    #[nonblocking]
    fn channels(&self) -> impl Iterator<Item = ndarray::ArrayView1<S>> {
        self.data.iter().map(move |&ch| unsafe {
            ArrayView1::from_shape_ptr(self.num_frames, ch as *const S as *mut S)
        })
    }
}

impl<'a, S: Sample> BlockWrite<S> for BlockViewStackedMut<'a, S> {
    #[nonblocking]
    fn sample_mut(&mut self, ch: u16, frame: usize) -> &mut S {
        unsafe { &mut *self.data[ch as usize].add(frame) }
    }

    #[nonblocking]
    fn channel_mut(&mut self, index: u16) -> ArrayViewMut1<S> {
        unsafe { ArrayViewMut1::from_shape_ptr(self.num_frames, self.data[index as usize]) }
    }

    #[nonblocking]
    fn channels_mut(&mut self) -> impl Iterator<Item = ndarray::ArrayViewMut1<S>> {
        let num_frames = self.num_frames;
        self.data
            .iter_mut()
            .map(move |ch| unsafe { ArrayViewMut1::from_shape_ptr(num_frames, *ch) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_slices() {
        let mut ch1 = vec![0.0_f32, 1.0, 2.0];
        let mut ch2 = vec![3.0_f32, 4.0, 5.0];
        let mut slices: Vec<&mut [f32]> = vec![&mut ch1, &mut ch2];
        let block = BlockViewStackedMut::from_slices(&mut slices);
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.sample(0, 1), 1.0);
        assert_eq!(block.sample(1, 2), 5.0);

        let ch_view = block.channel(0);
        assert_eq!(ch_view.len(), 3);
        let all_channels: Vec<_> = block.channels().collect();
        assert_eq!(all_channels.len(), 2);
    }

    #[test]
    fn test_from_ptr() {
        let mut ch1 = vec![10.0_f32, 11.0, 12.0];
        let mut ch2 = vec![13.0_f32, 14.0, 15.0];
        let mut slices: Vec<&mut [f32]> = vec![&mut ch1, &mut ch2];
        let ptrs: Vec<*mut f32> = slices.iter_mut().map(|c| c.as_mut_ptr()).collect();
        let block = BlockViewStackedMut::from_ptr(ptrs.as_ptr(), ptrs.len(), ch1.len());
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.sample(0, 0), 10.0);
        assert_eq!(block.sample(1, 2), 15.0);
    }

    #[test]
    fn test_write_ops() {
        let mut ch1 = vec![0.0_f32, 1.0, 2.0];
        let mut ch2 = vec![3.0_f32, 4.0, 5.0];
        let mut slices: Vec<&mut [f32]> = vec![&mut ch1, &mut ch2];
        let mut block = BlockViewStackedMut::from_slices(&mut slices);
        *block.sample_mut(0, 2) = 20.0;
        *block.sample_mut(1, 0) = 30.0;
        assert_eq!(block.sample(0, 2), 20.0);
        assert_eq!(block.sample(1, 0), 30.0);

        {
            let mut ch_view = block.channel_mut(0);
            ch_view[1] = 100.0;
        }
        assert_eq!(block.sample(0, 1), 100.0);

        for mut ch_view in block.channels_mut() {
            for s in ch_view.iter_mut() {
                *s += 1.0;
            }
        }

        assert_eq!(block.sample(0, 0), 1.0);
        assert_eq!(block.sample(1, 2), 6.0);
    }
}
