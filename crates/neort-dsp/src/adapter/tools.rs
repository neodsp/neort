use audio_blocks::{AudioBlock, AudioBlockMut, Ops, Stacked};
use neort_float::Float;

pub fn find_max_index_per_channel<F: Float>(audio: impl AudioBlock<F>) -> Vec<usize> {
    let mut maxima = Vec::with_capacity(audio.num_channels() as usize);
    for channel in audio.channels() {
        let index_of_max: Option<usize> = channel
            .enumerate()
            .map(|(index, value)| (index, value.abs()))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index);
        maxima.push(index_of_max.unwrap_or(0));
    }
    maxima
}

pub fn impulse_response<F: Float>(
    num_iterations: usize,
    num_channels: u16,
    num_frames: usize,
    mut process_fn: impl FnMut(&mut Stacked<F>),
) -> Stacked<F> {
    let mut impulse = Stacked::<F>::new(num_channels, num_frames);
    for ch in 0..num_channels {
        *impulse.sample_mut(ch, 0) = F::one();
    }

    let mut impulse_response = Stacked::new(num_channels, num_frames * num_iterations);
    process_fn(&mut impulse);

    for channel in 0..num_channels {
        for frame in 0..num_frames {
            *impulse_response.sample_mut(channel, frame) = impulse.sample(channel, frame);
        }
    }

    for _ in 1..num_iterations {
        impulse.clear();
        process_fn(&mut impulse);
        for channel in 0..num_channels {
            for frame in 0..num_frames {
                *impulse_response.sample_mut(channel, num_iterations * num_frames + frame) =
                    impulse.sample(channel, frame);
            }
        }
    }

    impulse_response
}
