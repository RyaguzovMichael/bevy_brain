#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Normalized(f32);
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct UNorm(f32);
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct SNorm(f32);

impl Normalized {
    #[inline]
    pub fn get(&self) -> f32 {
        self.0
    }
}

impl UNorm {
    #[inline]
    pub fn get(&self) -> f32 {
        self.0
    }

    #[inline]
    pub fn invert(&self) -> Self {
        Self(1.0 - self.0)
    }
}

impl SNorm {
    #[inline]
    pub fn get(&self) -> f32 {
        self.0
    }

    #[inline]
    pub fn invert(&self) -> Self {
        Self(-self.0)
    }
}

impl From<UNorm> for Normalized {
    fn from(value: UNorm) -> Self {
        Self(value.0)
    }
}

impl From<SNorm> for Normalized {
    fn from(value: SNorm) -> Self {
        Self(value.0)
    }
}

pub trait F32NormExt {
    fn to_unorm(self, max: f32) -> UNorm;
    fn to_snorm(self, range: f32) -> SNorm;
}

impl F32NormExt for f32 {
    #[inline]
    fn to_unorm(self, max: f32) -> UNorm {
        debug_assert!(max > 0.0);
        debug_assert!(!self.is_nan());

        UNorm(self.clamp(0.0, max) / max)
    }

    #[inline]
    fn to_snorm(self, range: f32) -> SNorm {
        debug_assert!(range > 0.0);
        debug_assert!(!self.is_nan());

        SNorm((self / range).clamp(-1.0, 1.0))
    }
}

pub trait BrainInput<const N: usize>: Send + Sync + 'static {
    fn fill(&self, _input: &mut [Normalized; N]) {
        debug_assert!(
            false,
            "BrainInput::fill is not implemented. It is required to feed data into neural network brains."
        );
    }
}

pub trait BrainOutput<const N: usize>: Send + Sync + 'static {
    fn apply(&mut self, _output: &[f32; N]) {
        debug_assert!(
            false,
            "BrainOutput::apply is not implemented. It is required to read output from neural network brains."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn convert_f32_to_unorm() {
        let unorm = 1.0.to_unorm(2.0);

        assert_eq!(0.5, unorm.get());
    }

    #[rstest]
    #[case(-0.000001)]
    #[case(-1.0)]
    #[case(f32::MIN)]
    fn when_convert_negative_f32_to_unorm_always_get_zero(#[case] input: f32) {
        let unorm = input.to_unorm(1.0);

        assert_eq!(0.0, unorm.get());
    }

    #[rstest]
    #[case(0.0, 1.0, 0.0)]
    #[case(1.0, 4.0, 0.25)]
    #[case(3.0, 4.0, 0.75)]
    fn to_unorm_normalizes_by_max(#[case] input: f32, #[case] max: f32, #[case] expected: f32) {
        let unorm = input.to_unorm(max);

        assert_eq!(expected, unorm.get());
    }

    #[rstest]
    #[case(1.0, 1.0)]
    #[case(5.0, 2.0)]
    #[case(f32::MAX, 1.0)]
    fn to_unorm_clamps_above_max_to_one(#[case] input: f32, #[case] max: f32,) {
        let unorm = input.to_unorm(max);

        assert_eq!(1.0, unorm.get());
    }

    #[rstest]
    #[case(0.0, 10.0, 0.0)]
    #[case(10.0, 10.0, 1.0)]
    #[case(-10.0, 10.0, -1.0)]
    #[case(5.0, 10.0, 0.5)]
    #[case(-5.0, 10.0, -0.5)]
    fn to_snorm_maps_range(#[case] input: f32, #[case] range: f32, #[case] expected: f32) {
        let snorm = input.to_snorm(range);

        assert_eq!(expected, snorm.get());
    }

    #[rstest]
    #[case(f32::MAX)]
    #[case(f32::INFINITY)]
    fn to_snorm_clamps_above_one(#[case] input: f32) {
        let snorm = input.to_snorm(1.0);

        assert_eq!(1.0, snorm.get());
    }

    #[rstest]
    #[case(f32::MIN)]
    #[case(f32::NEG_INFINITY)]
    fn to_snorm_clamps_below_minus_one(#[case] input: f32) {
        let snorm = input.to_snorm(1.0);

        assert_eq!(-1.0, snorm.get());
    }

    #[test]
    fn unorm_invert_returns_one_minus_value() {
        let unorm = 0.25.to_unorm(1.0);

        assert_eq!(0.75, unorm.invert().get());
    }

    #[test]
    fn unorm_invert_twice_returns_original() {
        let unorm = 0.3.to_unorm(1.0);

        assert_eq!(unorm.get(), unorm.invert().invert().get());
    }

    #[test]
    fn snorm_invert_negates_value() {
        let snorm = 0.4.to_snorm(1.0);

        assert_eq!(-0.4, snorm.invert().get());
    }

    #[test]
    fn snorm_invert_twice_returns_original() {
        let snorm = (-0.7).to_snorm(1.0);

        assert_eq!(snorm.get(), snorm.invert().invert().get());
    }

    #[test]
    fn normalized_from_unorm_keeps_value() {
        let unorm = 0.6.to_unorm(1.0);
        let normalized = Normalized::from(unorm);

        assert_eq!(0.6, normalized.get());
    }

    #[test]
    fn normalized_from_snorm_keeps_value() {
        let snorm = (-0.6).to_snorm(1.0);
        let normalized = Normalized::from(snorm);

        assert_eq!(-0.6, normalized.get());
    }

    #[test]
    fn default_values_are_zero() {
        assert_eq!(0.0, Normalized::default().get());
        assert_eq!(0.0, UNorm::default().get());
        assert_eq!(0.0, SNorm::default().get());
    }
}
