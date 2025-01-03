#![cfg_attr(not(feature = "std"), no_std)]

use core::cmp::PartialOrd;
use core::fmt::{Debug, Display};
use core::{f32, f64, ops::*};

pub trait FloatCore:
    Sized
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

    // Conversion
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;

    // Common methods
    fn is_nan(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;
    fn is_normal(self) -> bool;
    fn classify(self) -> core::num::FpCategory;
    fn to_degrees(self) -> Self;
    fn to_radians(self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
}

impl FloatCore for f32 {
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
    fn from_f64(value: f64) -> Self {
        value as Self
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    #[inline(always)]
    fn is_infinite(self) -> bool {
        self.is_infinite()
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    #[inline(always)]
    fn is_normal(self) -> bool {
        self.is_normal()
    }
    #[inline(always)]
    fn classify(self) -> core::num::FpCategory {
        self.classify()
    }
    #[inline(always)]
    fn to_degrees(self) -> Self {
        self.to_degrees()
    }
    #[inline(always)]
    fn to_radians(self) -> Self {
        self.to_radians()
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
}

impl FloatCore for f64 {
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
    fn from_f64(value: f64) -> Self {
        value
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    #[inline(always)]
    fn is_infinite(self) -> bool {
        self.is_infinite()
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    #[inline(always)]
    fn is_normal(self) -> bool {
        self.is_normal()
    }
    #[inline(always)]
    fn classify(self) -> core::num::FpCategory {
        self.classify()
    }
    #[inline(always)]
    fn to_degrees(self) -> Self {
        self.to_degrees()
    }
    #[inline(always)]
    fn to_radians(self) -> Self {
        self.to_radians()
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
}

pub trait Float: FloatCore {
    fn abs(self) -> Self;
    fn ceil(self) -> Self;
    fn floor(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn sqrt(self) -> Self;
    fn cbrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn powi(self, n: i32) -> Self;
}

impl Float for f32 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        self.ceil()
    }
    #[inline(always)]
    fn floor(self) -> Self {
        self.floor()
    }
    #[inline(always)]
    fn round(self) -> Self {
        self.round()
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        self.trunc()
    }
    #[inline(always)]
    fn fract(self) -> Self {
        self.fract()
    }
    #[inline(always)]
    fn exp(self) -> Self {
        self.exp()
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        self.exp2()
    }
    #[inline(always)]
    fn ln(self) -> Self {
        self.ln()
    }
    #[inline(always)]
    fn log2(self) -> Self {
        self.log2()
    }
    #[inline(always)]
    fn log10(self) -> Self {
        self.log10()
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline(always)]
    fn cbrt(self) -> Self {
        self.cbrt()
    }
    #[inline(always)]
    fn sin(self) -> Self {
        self.sin()
    }
    #[inline(always)]
    fn cos(self) -> Self {
        self.cos()
    }
    #[inline(always)]
    fn tan(self) -> Self {
        self.tan()
    }
    #[inline(always)]
    fn asin(self) -> Self {
        self.asin()
    }
    #[inline(always)]
    fn acos(self) -> Self {
        self.acos()
    }
    #[inline(always)]
    fn atan(self) -> Self {
        self.atan()
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        self.sinh()
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        self.cosh()
    }
    #[inline(always)]
    fn tanh(self) -> Self {
        self.tanh()
    }
    #[inline(always)]
    fn asinh(self) -> Self {
        self.asinh()
    }
    #[inline(always)]
    fn acosh(self) -> Self {
        self.acosh()
    }
    #[inline(always)]
    fn atanh(self) -> Self {
        self.atanh()
    }
    #[inline(always)]
    fn powf(self, n: Self) -> Self {
        self.powf(n)
    }
    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }
}

impl Float for f64 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        self.ceil()
    }
    #[inline(always)]
    fn floor(self) -> Self {
        self.floor()
    }
    #[inline(always)]
    fn round(self) -> Self {
        self.round()
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        self.trunc()
    }
    #[inline(always)]
    fn fract(self) -> Self {
        self.fract()
    }
    #[inline(always)]
    fn exp(self) -> Self {
        self.exp()
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        self.exp2()
    }
    #[inline(always)]
    fn ln(self) -> Self {
        self.ln()
    }
    #[inline(always)]
    fn log2(self) -> Self {
        self.log2()
    }
    #[inline(always)]
    fn log10(self) -> Self {
        self.log10()
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline(always)]
    fn cbrt(self) -> Self {
        self.cbrt()
    }
    #[inline(always)]
    fn sin(self) -> Self {
        self.sin()
    }
    #[inline(always)]
    fn cos(self) -> Self {
        self.cos()
    }
    #[inline(always)]
    fn tan(self) -> Self {
        self.tan()
    }
    #[inline(always)]
    fn asin(self) -> Self {
        self.asin()
    }
    #[inline(always)]
    fn acos(self) -> Self {
        self.acos()
    }
    #[inline(always)]
    fn atan(self) -> Self {
        self.atan()
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        self.sinh()
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        self.cosh()
    }
    #[inline(always)]
    fn tanh(self) -> Self {
        self.tanh()
    }
    #[inline(always)]
    fn asinh(self) -> Self {
        self.asinh()
    }
    #[inline(always)]
    fn acosh(self) -> Self {
        self.acosh()
    }
    #[inline(always)]
    fn atanh(self) -> Self {
        self.atanh()
    }
    #[inline(always)]
    fn powf(self, n: Self) -> Self {
        self.powf(n)
    }
    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }
}

pub trait IntoFloat {
    fn into_float<F: FloatCore>(self) -> F;
}

macro_rules! impl_into_float {
    ($($t:ty),*) => {
        $(
            impl IntoFloat for $t {
                #[inline(always)]
                fn into_float<F: FloatCore>(self) -> F {
                    F::from_f64(self as f64)
                }
            }
        )*
    }
}

impl_into_float!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);

pub trait FromFloat {
    fn from_float<F: FloatCore>(f: F) -> Self;
}

macro_rules! impl_from_float {
    ($($t:ty),*) => {
        $(
            impl FromFloat for $t {
                #[inline(always)]
                fn from_float<F: FloatCore>(f: F) -> Self {
                    f.to_f64() as Self
                }
            }
        )*
    }
}

impl_from_float!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float() {
        fn my_add<F: Float>(a: F, b: F) -> F {
            a + b
        }

        assert_eq!(my_add(1_f32, 2_f32), 3.0);

        fn my_pow<F: Float>(a: F) -> F {
            let b = usize::from_float(a);
            dbg!(b);
            a.powf(10.into_float())
        }

        assert_eq!(my_pow(2_f32), 1024.0);
    }
}
