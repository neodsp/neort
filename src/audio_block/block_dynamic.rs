use ndarray::ArrayView1;
use num::Float;

use super::BlockRead;

pub struct BlockDynamic<'a, F: Float> {
    data: &'a mut [*mut F],
    num_frames: usize,
}

impl<'a, F: Float> BlockDynamic<'a, F> {
    pub fn from(data: &'a mut [*mut F], samples: usize) -> Self {
        Self {
            data,
            num_frames: samples,
        }
    }

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

        BlockDynamic::from(pointers, num_frames)
    }

    pub fn channels_mut(&mut self) -> impl Iterator<Item = &mut [F]> {
        let samples = self.num_frames;
        self.data
            .iter_mut()
            .map(move |ch| unsafe { std::slice::from_raw_parts_mut(*ch, samples) })
    }
}

impl<'a, F: Float + 'static> BlockRead<F> for BlockDynamic<'a, F> {
    fn channel(&self, index: u16) -> ndarray::ArrayView1<F> {
        unsafe { ArrayView1::from_shape_ptr(self.num_frames, self.data[index as usize]) }
    }

    fn frame(&self, _index: usize) -> ndarray::ArrayView1<F> {
        panic!("Not implemented");
    }

    fn channels(&self) -> impl Iterator<Item = ndarray::ArrayView1<F>> {
        let samples = self.num_frames;
        self.data.iter().map(move |&ch| unsafe {
            ArrayView1::from_shape_ptr(samples, ch as *const F as *mut F)
        })
    }

    fn frames(&self) -> ndarray::iter::Lanes<F, ndarray::Dim<[usize; 1]>> {
        panic!("Not implemented");
    }

    fn view(&self) -> super::BlockView<F> {
        todo!()
    }

    fn view_slice(&self, range: std::ops::Range<usize>) -> super::BlockView<F> {
        todo!()
    }

    fn data(&self) -> ndarray::ArrayView2<F> {
        todo!()
    }

    fn raw_buffer(&self) -> &[F] {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_dynamic() {}
}
