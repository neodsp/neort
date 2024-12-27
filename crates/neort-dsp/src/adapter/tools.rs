use neort_blocks::{BlockHeap, BlockViewMut};

use crate::Float;

pub fn find_max_index<S: Float>(data: &[S]) -> usize {
    let index_of_max: Option<usize> = data
        .iter()
        .enumerate()
        .map(|(index, value)| (index, value.abs()))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index);
    index_of_max.unwrap()
}

pub fn impulse_response<S: Float>(
    num_iterations: usize,
    num_frames: usize,
    num_channels: usize,
    mut process_fn: impl FnMut(BlockViewMut<S>),
) -> Vec<S> {
    let mut impulse = BlockHeap::<S>::new(num_channels, num_frames);
    impulse.channel_mut(0)[0] = S::one();

    let mut impulse_response = Vec::new();
    process_fn(impulse.view_mut());

    for sample in impulse.channel(0).iter() {
        impulse_response.push(*sample);
    }

    for _ in 1..num_iterations {
        impulse.clear();
        process_fn(impulse.view_mut());
        for sample in impulse.channel(0).iter() {
            impulse_response.push(*sample);
        }
    }

    impulse_response
}
