use core::f64;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neort_float::{Float, FromGeneric, IntoGeneric};

fn floats_conversion<F: Float>() -> F {
    let pi = F::PI;
    let pi_int: usize = usize::from_f(pi);
    let a: F = pi_int.as_f();
    a + pi
}

fn num_traits_conversion<F: num_traits::Float>() -> F {
    let pi = F::from(f64::consts::PI).unwrap();
    let pi_int: usize = pi.to_usize().unwrap();
    let a: F = F::from(pi_int).unwrap();
    a + pi
}

pub fn float_bench(c: &mut Criterion) {
    c.bench_function("neort conversion f64", |b| {
        b.iter(|| black_box(floats_conversion::<f64>()))
    });

    c.bench_function("num_traits conversion f64", |b| {
        b.iter(|| black_box(num_traits_conversion::<f64>()))
    });

    c.bench_function("neort conversion f32", |b| {
        b.iter(|| black_box(floats_conversion::<f32>()))
    });

    c.bench_function("num_traits conversion f32", |b| {
        b.iter(|| black_box(num_traits_conversion::<f32>()))
    });
}

criterion_group!(benches, float_bench);
criterion_main!(benches);
