use num::Float;

use crate::audio_block::{Block, BlockRead, BlockWrite};

pub fn find_max_index<F: Float>(data: &[F]) -> usize {
    let index_of_max: Option<usize> = data
        .iter()
        .enumerate()
        .map(|(index, value)| (index, value.abs()))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index);
    index_of_max.unwrap()
}

pub fn impulse_response<F: Float>(
    num_iterations: usize,
    num_frames: usize,
    mut process_fn: impl FnMut(&mut Block<F>),
) -> Vec<F> {
    let mut impulse = Block::<F>::new(1, num_frames);
    impulse.channel_mut(0)[0] = F::one();

    let mut impulse_response = Vec::new();
    process_fn(&mut impulse);

    for sample in impulse.channel(0).iter() {
        impulse_response.push(*sample);
    }

    for _ in 1..num_iterations {
        impulse.clear();
        process_fn(&mut impulse);
        for sample in impulse.channel(0).iter() {
            impulse_response.push(*sample);
        }
    }

    impulse_response
}
