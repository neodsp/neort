#![cfg_attr(not(feature = "std"), no_std)]

use core::cmp::PartialOrd;
use core::fmt::{Debug, Display};
use core::{f32, f64, ops::*};

use num_traits::{FromPrimitive, Signed};

pub trait Float:
    num_traits::Float
    + Sized
    + Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Rem<Output = Self>
    + RemAssign
    + Neg<Output = Self>
    + Default
    + Debug
    + Display
    + Send
    + Sync
    + FromPrimitive
    + Signed
    + 'static
{
    // Constants
    const RADIX: u32;
    const MANTISSA_DIGITS: u32;
    const DIGITS: u32;
    const EPSILON: Self;
    const MIN: Self;
    const MAX: Self;
    const MIN_EXP: i32;
    const MAX_EXP: i32;
    const MIN_10_EXP: i32;
    const MAX_10_EXP: i32;
    const NAN: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;

    const PI: Self;
    const TAU: Self;
    const PHI: Self;
    const EGAMMA: Self;
    const FRAC_PI_2: Self;
    const FRAC_PI_3: Self;
    const FRAC_PI_4: Self;
    const FRAC_PI_6: Self;
    const FRAC_PI_8: Self;
    const FRAC_1_PI: Self;
    const FRAC_1_SQRT_PI: Self;
    const FRAC_1_SQRT_2PI: Self;
    const FRAC_2_PI: Self;
    const FRAC_2_SQRT_PI: Self;
    const SQRT_2: Self;
    const FRAC_1_SQRT_2: Self;
    const SQRT_3: Self;
    const FRAC_1_SQRT_3: Self;
    const E: Self;
    const LOG2_E: Self;
    const LOG2_10: Self;
    const LOG10_E: Self;
    const LOG10_2: Self;
    const LN_2: Self;
    const LN_10: Self;

    fn classify(self) -> core::num::FpCategory;

    // Conversion
    // Direct numeric conversions
    fn i8_as(value: i8) -> Self;
    fn i16_as(value: i16) -> Self;
    fn i32_as(value: i32) -> Self;
    fn i64_as(value: i64) -> Self;
    fn i128_as(value: i128) -> Self;
    fn isize_as(value: isize) -> Self;

    fn u8_as(value: u8) -> Self;
    fn u16_as(value: u16) -> Self;
    fn u32_as(value: u32) -> Self;
    fn u64_as(value: u64) -> Self;
    fn u128_as(value: u128) -> Self;
    fn usize_as(value: usize) -> Self;

    fn f32_as(value: f32) -> Self;
    fn f64_as(value: f64) -> Self;

    fn as_i8(self) -> i8;
    fn as_i16(self) -> i16;
    fn as_i32(self) -> i32;
    fn as_i64(self) -> i64;
    fn as_i128(self) -> i128;
    fn as_isize(self) -> isize;

    fn as_u8(self) -> u8;
    fn as_u16(self) -> u16;
    fn as_u32(self) -> u32;
    fn as_u64(self) -> u64;
    fn as_u128(self) -> u128;
    fn as_usize(self) -> usize;

    fn as_f32(self) -> f32;
    fn as_f64(self) -> f64;
}

impl Float for f32 {
    const RADIX: u32 = f32::RADIX;
    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;
    const DIGITS: u32 = f32::DIGITS;
    const EPSILON: Self = f32::EPSILON;
    const MIN: Self = f32::MIN;
    const MAX: Self = f32::MAX;
    const MIN_EXP: i32 = f32::MIN_EXP;
    const MAX_EXP: i32 = f32::MAX_EXP;
    const MIN_10_EXP: i32 = f32::MIN_10_EXP;
    const MAX_10_EXP: i32 = f32::MAX_10_EXP;
    const NAN: Self = f32::NAN;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;

    const PI: f32 = 3.14159265358979323846264338327950288_f32;
    const TAU: f32 = 6.28318530717958647692528676655900577_f32;
    const PHI: f32 = 1.618033988749894848204586834365638118_f32;
    const EGAMMA: f32 = 0.577215664901532860606512090082402431_f32;
    const FRAC_PI_2: f32 = 1.57079632679489661923132169163975144_f32;
    const FRAC_PI_3: f32 = 1.04719755119659774615421446109316763_f32;
    const FRAC_PI_4: f32 = 0.785398163397448309615660845819875721_f32;
    const FRAC_PI_6: f32 = 0.52359877559829887307710723054658381_f32;
    const FRAC_PI_8: f32 = 0.39269908169872415480783042290993786_f32;
    const FRAC_1_PI: f32 = 0.318309886183790671537767526745028724_f32;
    const FRAC_1_SQRT_PI: f32 = 0.564189583547756286948079451560772586_f32;
    const FRAC_1_SQRT_2PI: f32 = 0.398942280401432677939946059934381868_f32;
    const FRAC_2_PI: f32 = 0.636619772367581343075535053490057448_f32;
    const FRAC_2_SQRT_PI: f32 = 1.12837916709551257389615890312154517_f32;
    const SQRT_2: f32 = 1.41421356237309504880168872420969808_f32;
    const FRAC_1_SQRT_2: f32 = 0.707106781186547524400844362104849039_f32;
    const SQRT_3: f32 = 1.732050807568877293527446341505872367_f32;
    const FRAC_1_SQRT_3: f32 = 0.577350269189625764509148780501957456_f32;
    const E: f32 = 2.71828182845904523536028747135266250_f32;
    const LOG2_E: f32 = 1.44269504088896340735992468100189214_f32;
    const LOG2_10: f32 = 3.32192809488736234787031942948939018_f32;
    const LOG10_E: f32 = 0.434294481903251827651128918916605082_f32;
    const LOG10_2: f32 = 0.301029995663981195213738894724493027_f32;
    const LN_2: f32 = 0.693147180559945309417232121458176568_f32;
    const LN_10: f32 = 2.30258509299404568401799145468436421_f32;

    #[inline(always)]
    fn classify(self) -> core::num::FpCategory {
        self.classify()
    }

    #[inline(always)]
    fn i8_as(value: i8) -> Self {
        value as Self
    }
    #[inline(always)]
    fn i16_as(value: i16) -> Self {
        value as Self
    }
    #[inline(always)]
    fn i32_as(value: i32) -> Self {
        value as Self
    }
    #[inline(always)]
    fn i64_as(value: i64) -> Self {
        value as Self
    }
    #[inline(always)]
    fn i128_as(value: i128) -> Self {
        value as Self
    }
    #[inline(always)]
    fn isize_as(value: isize) -> Self {
        value as Self
    }

    #[inline(always)]
    fn u8_as(value: u8) -> Self {
        value as Self
    }
    #[inline(always)]
    fn u16_as(value: u16) -> Self {
        value as Self
    }
    #[inline(always)]
    fn u32_as(value: u32) -> Self {
        value as Self
    }
    #[inline(always)]
    fn u64_as(value: u64) -> Self {
        value as Self
    }
    #[inline(always)]
    fn u128_as(value: u128) -> Self {
        value as Self
    }
    #[inline(always)]
    fn usize_as(value: usize) -> Self {
        value as Self
    }

    #[inline(always)]
    fn f32_as(value: f32) -> Self {
        value
    }
    #[inline(always)]
    fn f64_as(value: f64) -> Self {
        value as Self
    }

    #[inline(always)]
    fn as_i8(self) -> i8 {
        self as i8
    }
    #[inline(always)]
    fn as_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn as_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn as_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn as_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn as_isize(self) -> isize {
        self as isize
    }

    #[inline(always)]
    fn as_u8(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn as_u16(self) -> u16 {
        self as u16
    }
    #[inline(always)]
    fn as_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn as_u64(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn as_u128(self) -> u128 {
        self as u128
    }
    #[inline(always)]
    fn as_usize(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn as_f32(self) -> f32 {
        self
    }
    #[inline(always)]
    fn as_f64(self) -> f64 {
        self as f64
    }
}

impl Float for f64 {
    const RADIX: u32 = f64::RADIX;
    const MANTISSA_DIGITS: u32 = f64::MANTISSA_DIGITS;
    const DIGITS: u32 = f64::DIGITS;
    const EPSILON: Self = f64::EPSILON;
    const MIN: Self = f64::MIN;
    const MAX: Self = f64::MAX;
    const MIN_EXP: i32 = f64::MIN_EXP;
    const MAX_EXP: i32 = f64::MAX_EXP;
    const MIN_10_EXP: i32 = f64::MIN_10_EXP;
    const MAX_10_EXP: i32 = f64::MAX_10_EXP;
    const NAN: Self = f64::NAN;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;

    const PI: f64 = 3.14159265358979323846264338327950288_f64;
    const TAU: f64 = 6.28318530717958647692528676655900577_f64;
    const PHI: f64 = 1.618033988749894848204586834365638118_f64;
    const EGAMMA: f64 = 0.577215664901532860606512090082402431_f64;
    const FRAC_PI_2: f64 = 1.57079632679489661923132169163975144_f64;
    const FRAC_PI_3: f64 = 1.04719755119659774615421446109316763_f64;
    const FRAC_PI_4: f64 = 0.785398163397448309615660845819875721_f64;
    const FRAC_PI_6: f64 = 0.52359877559829887307710723054658381_f64;
    const FRAC_PI_8: f64 = 0.39269908169872415480783042290993786_f64;
    const FRAC_1_PI: f64 = 0.318309886183790671537767526745028724_f64;
    const FRAC_1_SQRT_PI: f64 = 0.564189583547756286948079451560772586_f64;
    const FRAC_1_SQRT_2PI: f64 = 0.398942280401432677939946059934381868_f64;
    const FRAC_2_PI: f64 = 0.636619772367581343075535053490057448_f64;
    const FRAC_2_SQRT_PI: f64 = 1.12837916709551257389615890312154517_f64;
    const SQRT_2: f64 = 1.41421356237309504880168872420969808_f64;
    const FRAC_1_SQRT_2: f64 = 0.707106781186547524400844362104849039_f64;
    const SQRT_3: f64 = 1.732050807568877293527446341505872367_f64;
    const FRAC_1_SQRT_3: f64 = 0.577350269189625764509148780501957456_f64;
    const E: f64 = 2.71828182845904523536028747135266250_f64;
    const LOG2_10: f64 = 3.32192809488736234787031942948939018_f64;
    const LOG2_E: f64 = 1.44269504088896340735992468100189214_f64;
    const LOG10_2: f64 = 0.301029995663981195213738894724493027_f64;
    const LOG10_E: f64 = 0.434294481903251827651128918916605082_f64;
    const LN_2: f64 = 0.693147180559945309417232121458176568_f64;
    const LN_10: f64 = 2.30258509299404568401799145468436421_f64;

    #[inline(always)]
    fn classify(self) -> core::num::FpCategory {
        self.classify()
    }

    #[inline(always)]
    fn i8_as(value: i8) -> Self {
        value as Self
    }
    #[inline(always)]
    fn i16_as(value: i16) -> Self {
        value as Self
    }
    #[inline(always)]
    fn i32_as(value: i32) -> Self {
        value as Self
    }
    #[inline(always)]
    fn i64_as(value: i64) -> Self {
        value as Self
    }
    #[inline(always)]
    fn i128_as(value: i128) -> Self {
        value as Self
    }
    #[inline(always)]
    fn isize_as(value: isize) -> Self {
        value as Self
    }

    #[inline(always)]
    fn u8_as(value: u8) -> Self {
        value as Self
    }
    #[inline(always)]
    fn u16_as(value: u16) -> Self {
        value as Self
    }
    #[inline(always)]
    fn u32_as(value: u32) -> Self {
        value as Self
    }
    #[inline(always)]
    fn u64_as(value: u64) -> Self {
        value as Self
    }
    #[inline(always)]
    fn u128_as(value: u128) -> Self {
        value as Self
    }
    #[inline(always)]
    fn usize_as(value: usize) -> Self {
        value as Self
    }

    #[inline(always)]
    fn f32_as(value: f32) -> Self {
        value as Self
    }
    #[inline(always)]
    fn f64_as(value: f64) -> Self {
        value
    }

    #[inline(always)]
    fn as_i8(self) -> i8 {
        self as i8
    }
    #[inline(always)]
    fn as_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn as_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn as_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn as_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn as_isize(self) -> isize {
        self as isize
    }

    #[inline(always)]
    fn as_u8(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn as_u16(self) -> u16 {
        self as u16
    }
    #[inline(always)]
    fn as_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn as_u64(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn as_u128(self) -> u128 {
        self as u128
    }
    #[inline(always)]
    fn as_usize(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn as_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn as_f64(self) -> f64 {
        self
    }
}

pub trait IntoGeneric<T: Copy> {
    #[allow(clippy::wrong_self_convention)]
    fn as_f<F: Float>(self) -> F;
}

// Individual implementations for each type instead of using a match
macro_rules! impl_into_generic {
    ($t:ty, $conv:ident) => {
        impl IntoGeneric<$t> for $t {
            #[inline(always)]
            fn as_f<F: Float>(self) -> F {
                F::$conv(self)
            }
        }
    };
}

// Generate specific implementations for each type
impl_into_generic!(i8, i8_as);
impl_into_generic!(i16, i16_as);
impl_into_generic!(i32, i32_as);
impl_into_generic!(i64, i64_as);
impl_into_generic!(i128, i128_as);
impl_into_generic!(isize, isize_as);
impl_into_generic!(u8, u8_as);
impl_into_generic!(u16, u16_as);
impl_into_generic!(u32, u32_as);
impl_into_generic!(u64, u64_as);
impl_into_generic!(u128, u128_as);
impl_into_generic!(usize, usize_as);
impl_into_generic!(f32, f32_as);
impl_into_generic!(f64, f64_as);

// Update FromFloats trait to be generic over the target type
pub trait FromGeneric<T> {
    fn from_f<F: Float>(f: F) -> T;
}

// Individual implementations for each type
macro_rules! impl_from_floats {
    ($t:ty, $conv:ident) => {
        impl FromGeneric<$t> for $t {
            #[inline(always)]
            fn from_f<F: Float>(f: F) -> Self {
                f.$conv()
            }
        }
    };
}

// Generate specific implementations for each type
impl_from_floats!(i8, as_i8);
impl_from_floats!(i16, as_i16);
impl_from_floats!(i32, as_i32);
impl_from_floats!(i64, as_i64);
impl_from_floats!(i128, as_i128);
impl_from_floats!(isize, as_isize);
impl_from_floats!(u8, as_u8);
impl_from_floats!(u16, as_u16);
impl_from_floats!(u32, as_u32);
impl_from_floats!(u64, as_u64);
impl_from_floats!(u128, as_u128);
impl_from_floats!(usize, as_usize);
impl_from_floats!(f32, as_f32);
impl_from_floats!(f64, as_f64);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f32::consts::PI as STD_PI_F32;
    use core::f64::consts::PI as STD_PI;

    // Define epsilon constants for different precision levels
    const EPSILON_F64: f64 = 1e-10;
    const EPSILON_F32: f32 = 1e-6;

    // Helper function to test a type that implements Floats
    fn test_float_implementation<T: Float>()
    where
        T: approx::RelativeEq,
        T::Epsilon: Copy,
    {
        let epsilon = 1e-6;
        // Test constants
        assert_relative_eq!(T::PI.as_f64(), 3.141592653589793, epsilon = epsilon);
        assert_relative_eq!(T::E.as_f64(), 2.718281828459045, epsilon = epsilon);
        assert_relative_eq!(T::PHI.as_f64(), 1.618033988749895, epsilon = epsilon);
        assert_relative_eq!(T::TAU.as_f64(), T::PI.as_f64() * 2.0, epsilon = epsilon);

        // Test basic arithmetic
        let x = T::f64_as(2.5);
        let y = T::f64_as(1.5);
        assert_relative_eq!((x + y).as_f64(), 4.0, epsilon = epsilon);
        assert_relative_eq!((x - y).as_f64(), 1.0, epsilon = epsilon);
        assert_relative_eq!((x * y).as_f64(), 3.75, epsilon = epsilon);
        assert_relative_eq!((x / y).as_f64(), 1.6666666666666667, epsilon = epsilon);

        // Test classification methods
        assert!(!x.is_nan());
        assert!(T::NAN.is_nan());
        assert!(T::INFINITY.is_infinite());
        assert!(x.is_finite());
        assert!(x.is_normal());

        // Test conversion methods
        assert_relative_eq!(
            T::f64_as(STD_PI).to_degrees().as_f64(),
            180.0,
            epsilon = epsilon
        );
        assert_relative_eq!(
            T::f64_as(180.0).to_radians().as_f64(),
            STD_PI,
            epsilon = epsilon
        );

        // Test rounding functions
        let z = T::f64_as(3.7);
        assert_relative_eq!(z.ceil().as_f64(), 4.0, epsilon = epsilon);
        assert_relative_eq!(z.floor().as_f64(), 3.0, epsilon = epsilon);
        assert_relative_eq!(z.round().as_f64(), 4.0, epsilon = epsilon);
        assert_relative_eq!(z.trunc().as_f64(), 3.0, epsilon = epsilon);
        assert_relative_eq!(z.fract().as_f64(), 0.7, epsilon = epsilon);

        // Test exponential functions
        let w = T::f64_as(2.0);
        assert_relative_eq!(w.exp().as_f64(), 7.389056098930650, epsilon = epsilon);
        assert_relative_eq!(w.exp2().as_f64(), 4.0, epsilon = epsilon);
        assert_relative_eq!(w.ln().as_f64(), 0.693147180559945, epsilon = epsilon);
        assert_relative_eq!(w.log2().as_f64(), 1.0, epsilon = epsilon);
        assert_relative_eq!(w.log10().as_f64(), 0.301029995663981, epsilon = epsilon);

        // Test power functions
        assert_relative_eq!(w.sqrt().as_f64(), 1.414213562373095, epsilon = epsilon);
        assert_relative_eq!(w.cbrt().as_f64(), 1.259921049894873, epsilon = epsilon);
        assert_relative_eq!(w.powf(T::f64_as(3.0)).as_f64(), 8.0, epsilon = epsilon);
        assert_relative_eq!(w.powi(3).as_f64(), 8.0, epsilon = epsilon);

        // Test trigonometric functions
        let angle = T::PI / T::f64_as(6.0); // 30 degrees
        assert_relative_eq!(angle.sin().as_f64(), 0.5, epsilon = epsilon);
        assert_relative_eq!(angle.cos().as_f64(), 0.866025403784439, epsilon = epsilon);
        assert_relative_eq!(angle.tan().as_f64(), 0.577350269189626, epsilon = epsilon);

        // Test inverse trigonometric functions
        let v = T::f64_as(0.5);
        assert_relative_eq!(v.asin().as_f64(), 0.523598775598299, epsilon = epsilon);
        assert_relative_eq!(v.acos().as_f64(), 1.047197551196598, epsilon = epsilon);
        assert_relative_eq!(v.atan().as_f64(), 0.463647609000806, epsilon = epsilon);

        // Test hyperbolic functions
        assert_relative_eq!(v.sinh().as_f64(), 0.521095305493747, epsilon = epsilon);
        assert_relative_eq!(v.cosh().as_f64(), 1.127625965, epsilon = epsilon);
        assert_relative_eq!(v.tanh().as_f64(), 0.462117157260010, epsilon = epsilon);
        assert_relative_eq!(v.asinh().as_f64(), 0.481211825059603, epsilon = epsilon);
        assert_relative_eq!(
            T::f64_as(2.0).acosh().as_f64(),
            1.316957896924817,
            epsilon = epsilon
        );
        assert_relative_eq!(
            T::f64_as(0.5).atanh().as_f64(),
            0.549306144334055,
            epsilon = epsilon
        );

        // Test min/max
        assert_relative_eq!(
            T::f64_as(2.5).max(T::f64_as(1.5)).as_f64(),
            2.5,
            epsilon = epsilon
        );
        assert_relative_eq!(
            T::f64_as(2.5).min(T::f64_as(1.5)).as_f64(),
            1.5,
            epsilon = epsilon
        );

        // Test absolute value
        assert_relative_eq!(T::f64_as(-2.5).abs().as_f64(), 2.5, epsilon = epsilon);
    }

    #[test]
    fn test_f32_implementation() {
        test_float_implementation::<f32>();
    }

    #[test]
    fn test_f64_implementation() {
        test_float_implementation::<f64>();
    }

    // Additional specific tests for edge cases
    #[test]
    fn test_edge_cases() {
        // Test infinity
        assert!(f64::INFINITY.is_infinite());
        assert!(f32::INFINITY.is_infinite());

        // Test NaN
        assert!(f64::NAN.is_nan());
        assert!(f32::NAN.is_nan());

        // Test minimal values
        assert!(f64::MIN.is_finite());
        assert!(f32::MIN.is_finite());

        // Test maximum values
        assert!(f64::MAX.is_finite());
        assert!(f32::MAX.is_finite());
    }

    // Test specific mathematical constants
    #[test]
    fn test_mathematical_constants() {
        // Test PI accuracy for both f32 and f64
        assert_relative_eq!(f64::PI, STD_PI, epsilon = EPSILON_F64);
        assert_relative_eq!(f32::PI, STD_PI_F32, epsilon = EPSILON_F32);

        // Test other important constants
        assert_relative_eq!(f64::E, core::f64::consts::E, epsilon = EPSILON_F64);
        assert_relative_eq!(f32::E, core::f32::consts::E, epsilon = EPSILON_F32);

        assert_relative_eq!(f64::LN_2, core::f64::consts::LN_2, epsilon = EPSILON_F64);
        assert_relative_eq!(f32::LN_2, core::f32::consts::LN_2, epsilon = EPSILON_F32);

        assert_relative_eq!(f64::LN_10, core::f64::consts::LN_10, epsilon = EPSILON_F64);
        assert_relative_eq!(f32::LN_10, core::f32::consts::LN_10, epsilon = EPSILON_F32);
    }

    fn float_function<F: Float>(_: F) {}

    fn from_float<F: Float>(f: F) {
        let _ = i8::from_f(f);
        let _ = i16::from_f(f);
        let _ = i32::from_f(f);
        let _ = i64::from_f(f);
        let _ = i128::from_f(f);
        let _ = isize::from_f(f);
        let _ = u8::from_f(f);
        let _ = u16::from_f(f);
        let _ = u32::from_f(f);
        let _ = u64::from_f(f);
        let _ = u128::from_f(f);
        let _ = usize::from_f(f);
        let _ = f32::from_f(f);
        let _ = f64::from_f(f);
    }

    #[test]
    fn test_conversions() {
        let v = 0_i8;
        float_function::<f32>(v.as_f());
        float_function::<f64>(v.as_f());

        let v = 0_i16;
        float_function::<f32>(v.as_f());
        float_function::<f64>(v.as_f());

        let v = 0_i32;
        float_function::<f32>(v.as_f());
        float_function::<f64>(v.as_f());

        let v = 0_i64;
        float_function::<f32>(v.as_f());
        float_function::<f64>(v.as_f());

        let v = 0_i128;
        float_function::<f32>(v.as_f());
        float_function::<f64>(v.as_f());

        let v = 0_isize;
        float_function::<f32>(v.as_f());
        float_function::<f64>(v.as_f());

        let v = 0_u8;
        float_function::<f32>(v.as_f());
        float_function::<f64>(v.as_f());

        let v = 0_u16;
        float_function::<f32>(v.as_f());
        float_function::<f64>(v.as_f());

        let v = 0_u32;
        float_function::<f32>(v.as_f());
        float_function::<f64>(v.as_f());

        let v = 0_u64;
        float_function::<f32>(v.as_f());
        float_function::<f64>(v.as_f());

        let v = 0_u128;
        float_function::<f32>(v.as_f());
        float_function::<f64>(v.as_f());

        let v = 0_usize;
        float_function::<f32>(v.as_f());
        float_function::<f64>(v.as_f());

        let v = 0_f32;
        float_function::<f32>(v.as_f());
        float_function::<f64>(v.as_f());

        let v = 0_f64;
        float_function::<f32>(v.as_f());
        float_function::<f64>(v.as_f());

        let v = 0_f64;
        float_function::<f32>(v.as_f());
        float_function::<f64>(v.as_f());

        from_float(0_f32);
        from_float(0_f64);
    }
}
