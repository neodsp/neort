# Floats

The reason for this generic float crate is that num_traits::Float always was difficult to use.
For example you have to use `Floats::from(0.5).unwrap()` instead of just `0.5`.
On top this adds overhead if you have to unwrap for every number in a hot loop.

This crate provides:
- more functionality of floating points (e.g. `Debug` and `Default`)
- more operators (e.g. `AddAssign`)
- all constants in the std library (e.g. `Floats::PI`)
- conversions with less overhead and wihtout unwraps (e.g. `10.into_floats()` or `usize::from_floats(my_generic_float)`)
