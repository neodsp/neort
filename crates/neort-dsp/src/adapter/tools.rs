use neort_blocks::{BlockHeap, BlockViewMut};
use neort_float::Float;
use num_integer::lcm;

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
    num_channels: usize,
    mut process_fn: impl FnMut(BlockViewMut<F>),
) -> Vec<F> {
    let mut impulse = BlockHeap::<F>::new(num_channels, num_frames);
    impulse.channel_mut(0)[0] = F::one();

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

// calculation of frame-shift according to
// Stéphane Letz. Callback adaptation techniques. [Technical Report] GRAME. 2001. hal-02158912
pub fn calculate_frame_shift(host_buffer_len: usize, user_buffer_len: usize) -> usize {
    let mut res = 0;
    for i in (host_buffer_len..lcm(host_buffer_len, user_buffer_len)).step_by(host_buffer_len) {
        res = res.max(i % user_buffer_len);
    }
    res
}
