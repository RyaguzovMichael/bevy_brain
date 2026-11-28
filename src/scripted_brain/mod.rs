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
