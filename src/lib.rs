use signals::{BrainInput, BrainOutput};

pub mod scripted_brain;
mod signals;

pub mod prelude {
    pub use crate::signals::*;
    pub use crate::Brain;
}

pub trait Brain<const IN: usize, const OUT: usize, I, O>: Send + Sync + 'static
where
    I: BrainInput<IN>,
    O: BrainOutput<OUT>,
{
    fn think(&self, input: &I) -> O;
}
