use core::f64;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neort_floats::{Floats, FromFloats, IntoFloats};

fn floats_conversion<F: Floats>() -> F {
    let pi = F::PI;
    let pi_int: usize = usize::from_floats(pi);
    let a: F = pi_int.into_floats();
    a + pi
}

fn num_traits_conversion<F: num_traits::Float>() -> F {
    let pi = F::from(f64::consts::PI).unwrap();
    let pi_int: usize = pi.to_usize().unwrap();
    let a: F = F::from(pi_int).unwrap();
    a + pi
}

pub fn floats_bench(c: &mut Criterion) {
    c.bench_function("floats_conversion", |b| {
        b.iter(|| black_box(floats_conversion::<f64>()))
    });

    c.bench_function("num_traits_conversion", |b| {
        b.iter(|| black_box(num_traits_conversion::<f64>()))
    });
}

criterion_group!(benches, floats_bench);
criterion_main!(benches);
