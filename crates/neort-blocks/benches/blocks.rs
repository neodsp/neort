use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neort_blocks::BlockHeap;

pub fn blocks(c: &mut Criterion) {
    let mut block = BlockHeap::<f32>::new(2, 512);
    c.bench_function("view", |b| b.iter(|| black_box(block.view())));

    c.bench_function("channel iter", |b| {
        b.iter(|| block.channels_mut().for_each(|ch| ch.fill(2.0)))
    });

    c.bench_function("block index per sample", |b| {
        b.iter(|| {
            for channel in 0..block.num_channels() {
                for frame in 0..block.num_frames() {
                    block[[channel, frame]] *= 2.0;
                }
            }
        })
    });

    c.bench_function("block index per sample frames first", |b| {
        b.iter(|| {
            for frame in 0..block.num_frames() {
                for channel in 0..block.num_channels() {
                    block[[channel, frame]] *= 2.0;
                }
            }
        })
    });

    c.bench_function("block index per block", |b| {
        b.iter(|| {
            for channel in 0..block.num_channels() {
                for frame in 0..block.num_frames() {
                    block[channel][frame] *= 2.0;
                }
            }
        })
    });
}

criterion_group!(benches, blocks);
criterion_main!(benches);
