use crate::{
    audio_block::{Block, BlockRead},
    Sample,
};

pub fn find_max_index<S: Sample>(data: &[S]) -> usize {
    let index_of_max: Option<usize> = data
        .iter()
        .enumerate()
        .map(|(index, value)| (index, value.abs()))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index);
    index_of_max.unwrap()
}

pub fn impulse_response<S: Sample>(
    num_iterations: usize,
    num_frames: usize,
    num_channels: u16,
    mut process_fn: impl FnMut(&mut Block<S>),
) -> Vec<S> {
    let mut impulse = Block::<S>::new(num_channels, num_frames);
    impulse.channel_mut(0)[0] = S::one();

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
