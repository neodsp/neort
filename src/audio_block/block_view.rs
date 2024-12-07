use ndarray::{iter::Lanes, ArrayView1, ArrayView2, Dim, ShapeBuilder};
use num::Float;

use super::{Block, BlockRead, BufferLayout};

#[derive(Debug, Clone, PartialEq)]
pub struct BlockView<'a, F: Float> {
    data: ArrayView2<'a, F>,
}

impl<'a, F: Float> BlockView<'a, F> {
    #[rtsan::nonblocking]
    #[inline(always)]
    pub fn from_buffer(
        buffer: &'a [F],
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

    #[rtsan::nonblocking]
    #[inline(always)]
    pub fn from_array_view(view: ArrayView2<'a, F>) -> Self {
        Self { data: view }
    }

    pub fn to_owned(&self, force_sequential_layout: bool) -> Block<F> {
        if force_sequential_layout {
            Block::from_array(self.data.as_standard_layout().to_owned())
        } else {
            Block::from_array(self.data.to_owned())
        }
    }
}

impl<F: Float> BlockRead<F> for BlockView<'_, F> {
    #[rtsan::nonblocking]
    #[inline(always)]
    fn channel(&self, index: u16) -> ArrayView1<F> {
        self.data.row(index as usize)
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn frame(&self, index: u32) -> ArrayView1<F> {
        self.data.column(index as usize)
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn channels(&self) -> Lanes<F, Dim<[usize; 1]>> {
        self.data.rows()
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn frames(&self) -> Lanes<F, Dim<[usize; 1]>> {
        self.data.columns()
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn view(&self) -> BlockView<F> {
        BlockView::from_array_view(self.data.view())
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn raw_buffer(&self) -> &[F] {
        self.data.as_slice_memory_order().unwrap()
    }

    #[rtsan::nonblocking]
    #[inline(always)]
    fn data(&self) -> ndarray::ArrayView2<F> {
        self.data.view()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, aview1, aview2};

    use super::*;

    #[test]
    fn block_view_sequential() {
        let block = BlockView::from_buffer(
            &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2],
            2,
            3,
            BufferLayout::Sequential,
        );

        assert_eq!(
            block.to_owned(false),
            Block::from_array(array![[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]])
        );

        assert_eq!(
            block.to_owned(false).raw_buffer(),
            &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2]
        );

        // Block Read
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.layout(), BufferLayout::Sequential);

        // sample
        assert_eq!(block.sample(0, 1), 0.1);
        assert_eq!(block.sample(1, 2), 1.2);

        // fn channel(&self, index: u16) -> ArrayView1<Sample>;
        assert_eq!(block.channel(0), aview1(&[0.0, 0.1, 0.2]));
        assert_eq!(block.channel(1), aview1(&[1.0, 1.1, 1.2]));
        // fn frame(&self, index: u32) -> ArrayView1<Sample>;
        assert_eq!(block.frame(0), aview1(&[0.0, 1.0]));
        assert_eq!(block.frame(1), aview1(&[0.1, 1.1]));
        assert_eq!(block.frame(2), aview1(&[0.2, 1.2]));
        // fn channels(&self) -> Lanes<Sample, Dim<[usize; 1]>>;
        for (ch, channel) in block.channels().into_iter().enumerate() {
            if ch == 0 {
                assert_eq!(channel, aview1(&[0.0, 0.1, 0.2]));
            } else if ch == 1 {
                assert_eq!(channel, aview1(&[1.0, 1.1, 1.2]));
            }
        }
        // fn frames(&self) -> Lanes<Sample, Dim<[usize; 1]>>;
        for (fr, frame) in block.frames().into_iter().enumerate() {
            if fr == 0 {
                assert_eq!(frame, aview1(&[0.0, 1.0]));
            } else if fr == 1 {
                assert_eq!(frame, aview1(&[0.1, 1.1]));
            } else if fr == 2 {
                assert_eq!(frame, aview1(&[0.2, 1.2]));
            }
        }
        // fn view(&self) -> BlockView<Sample>;
        assert_eq!(
            block.view(),
            BlockView::from_array_view(aview2(&[[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]))
        );
        // fn raw_buffer(&self) -> &[Sample];
        assert_eq!(block.raw_buffer(), &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2]);
    }

    #[test]
    fn block_view_interleaved() {
        let block = BlockView::from_buffer(
            &[0.0, 1.0, 0.1, 1.1, 0.2, 1.2],
            2,
            3,
            BufferLayout::Interleaved,
        );

        assert_eq!(
            block.to_owned(false),
            Block::from_array(array![[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]])
        );

        assert_eq!(
            block.to_owned(false).raw_buffer(),
            &[0.0, 1.0, 0.1, 1.1, 0.2, 1.2]
        );

        assert_eq!(
            block.to_owned(true).raw_buffer(),
            &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2]
        );

        assert_eq!(block.layout(), BufferLayout::Interleaved);
        assert_eq!(
            block.view(),
            BlockView::from_array_view(aview2(&[[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]))
        );
        assert_eq!(block.raw_buffer(), &[0.0, 1.0, 0.1, 1.1, 0.2, 1.2]);
    }
}
