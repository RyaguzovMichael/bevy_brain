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
    fn fill(&self, input: &mut [Normalized; N]);
}

pub trait BrainOutput<const N: usize>: Send + Sync + 'static {
    fn apply(&mut self, output: &[f32; N]);
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
}
