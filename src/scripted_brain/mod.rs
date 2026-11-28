use std::marker::PhantomData;

use crate::{
    Brain,
    signals::{BrainInput, BrainOutput},
};

pub struct ScriptedBrain<const IN: usize, const OUT: usize, I, O, F>
where
    I: BrainInput<IN>,
    O: BrainOutput<OUT>,
    F: Fn(&I) -> O + Send + Sync + 'static,
{
    calculation: F,
    _marker: PhantomData<fn(&I) -> O>,
}

impl<const IN: usize, const OUT: usize,I, O, F> ScriptedBrain<IN, OUT,I, O, F>
where
    I: BrainInput<IN>,
    O: BrainOutput<OUT>,
    F: Fn(&I) -> O + Send + Sync + 'static,
{
    pub fn create(calculation: F) -> ScriptedBrain<IN, OUT, I, O, F> {
        ScriptedBrain {
            calculation,
            _marker: PhantomData,
        }
    }
}

impl<const IN: usize, const OUT: usize, I, O, F> Brain<IN, OUT, I, O>
    for ScriptedBrain<IN, OUT, I, O, F>
where
    I: BrainInput<IN>,
    O: BrainOutput<OUT>,
    F: Fn(&I) -> O + Send + Sync + 'static,
{
    fn think(&self, input: &I) -> O {
        (self.calculation)(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signals::{BrainInput, BrainOutput};

    struct TestInput {
        value: f32,
    }

    impl BrainInput<1> for TestInput {}

    #[derive(Debug, PartialEq)]
    struct TestOutput {
        doubled: f32,
    }

    impl BrainOutput<1> for TestOutput {}

    #[test]
    fn think_returns_closure_result() {
        let brain = ScriptedBrain::create(|input: &TestInput| TestOutput {
            doubled: input.value * 2.0,
        });

        let output = brain.think(&TestInput { value: 21.0 });

        assert_eq!(TestOutput { doubled: 42.0 }, output);
    }

    #[test]
    fn think_can_be_called_multiple_times() {
        let brain = ScriptedBrain::create(|input: &TestInput| TestOutput {
            doubled: input.value + 1.0,
        });

        assert_eq!(1.0, brain.think(&TestInput { value: 0.0 }).doubled);
        assert_eq!(2.0, brain.think(&TestInput { value: 1.0 }).doubled);
    }

    #[test]
    fn think_uses_latest_input() {
        let brain = ScriptedBrain::create(|input: &TestInput| TestOutput {
            doubled: input.value,
        });

        let first = brain.think(&TestInput { value: 1.0 });
        let second = brain.think(&TestInput { value: 9.0 });

        assert_eq!(1.0, first.doubled);
        assert_eq!(9.0, second.doubled);
    }
}
