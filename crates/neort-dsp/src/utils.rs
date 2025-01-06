use neort_float::{Float, IntoGeneric};

#[inline(always)]
pub fn db_to_gain<F: Float>(value: F) -> F {
    10.as_f::<F>().powf(value / 20.as_f())
}

#[inline(always)]
pub fn gain_to_db<F: Float>(value: F) -> F {
    20.as_f::<F>() * value.log10()
}

/// maps a value from one range into another range
#[inline(always)]
pub fn map<F: Float>(source: F, source_min: F, source_max: F, target_min: F, target_max: F) -> F {
    assert_ne!(source_min, source_max);
    target_min + ((target_max - target_min) * (source - source_min)) / (source_max - source_min)
}

/// maps a linear value between 0.0 and 1.0 into a logarithmic value range
#[inline(always)]
pub fn map_to_log10<F: Float>(value_0_to_1: F, log_range_min: F, log_range_max: F) -> F {
    assert!(log_range_min.is_positive());
    assert!(log_range_max.is_positive());
    let log_min = log_range_min.log10();
    let log_max = log_range_max.log10();
    10.as_f::<F>()
        .powf(value_0_to_1 * (log_max - log_min) + log_min)
}

/// maps a logarithmic value range into a linear value between 0.0 and 1.0
#[inline(always)]
pub fn map_from_log10<F: Float>(value_in_log_range: F, log_range_min: F, log_range_max: F) -> F {
    assert!(log_range_min.is_positive());
    assert!(log_range_max.is_positive());
    let min = log_range_min.log10();
    let max = log_range_max.log10();
    let value = value_in_log_range.log10();
    (value - min) / (max - min)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_to_gain() {
        assert!((db_to_gain(0.0f32) - 1.0f32).abs() <= 1e-6);
        assert!((db_to_gain(6.0f32) - 1.9952623f32).abs() <= 1e-6);
        assert!((db_to_gain(-6.0f32) - 0.50118726f32).abs() <= 1e-6);
    }

    #[test]
    fn test_gain_to_db() {
        assert!((gain_to_db(1.0f32) - 0.0f32).abs() <= 1e-6);
        assert!((gain_to_db(2.0f32) - 6.0206003f32).abs() <= 1e-6);
        assert!((gain_to_db(0.5f32) - -6.0206003f32).abs() <= 1e-6);
    }

    #[test]
    fn test_map() {
        assert_eq!(map(0.5f32, 0.0, 1.0, 0.0, 100.0), 50.0);
        assert_eq!(map(0.25f32, 0.0, 1.0, -1.0, 1.0), -0.5);
        assert_eq!(map(75f32, 0.0, 100.0, 0.0, 1.0), 0.75);
    }

    #[test]
    #[should_panic]
    fn test_map_invalid_range() {
        map(0.5f32, 1.0, 1.0, 0.0, 1.0);
    }

    #[test]
    fn test_map_to_log10() {
        assert!((map_to_log10(0.0f32, 20.0, 20000.0) - 20.0).abs() <= 1e-5);
        assert!((map_to_log10(0.5f32, 20.0, 20000.0) - 632.45575).abs() <= 1e-5);
        assert!((map_to_log10(1.0f32, 20.0, 20000.0) - 20000.0).abs() <= 1e-2);
    }

    #[test]
    #[should_panic]
    fn test_map_to_log10_invalid_range() {
        map_to_log10(0.5f32, -1.0, 1.0);
    }

    #[test]
    fn test_map_from_log10() {
        assert!((map_from_log10(20.0f32, 20.0, 20000.0) - 0.0).abs() <= 1e-6);
        assert!((map_from_log10(632.4555f32, 20.0, 20000.0) - 0.5).abs() <= 1e-3);
        assert!((map_from_log10(20000.0f32, 20.0, 20000.0) - 1.0).abs() <= 1e-6);
    }

    #[test]
    #[should_panic]
    fn test_map_from_log10_invalid_range() {
        map_from_log10(0.5f32, -1.0, 1.0);
    }
}
