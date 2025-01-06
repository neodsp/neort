# Generic Float

The reason for this generic float crate is that num_traits::Float always was difficult to use.
For example you have to use `Floats::from(0.5).unwrap()` instead of just `0.5`.
This crate adds the `as_f` functions that are faster (e.g. `0.as_f()`) and more convenient to use.
On top this adds overhead if you have to unwrap for every number in a hot loop.

This crate provides:
- more functionality of floating points (e.g. `Debug` and `Default`)
- more operators (e.g. `AddAssign`)
- all constants in the std library (e.g. `GenericFloat::PI`)
- conversions with less overhead and wihtout unwraps (e.g. `10.as_f()` or `usize::from_f(my_generic_float)`)
