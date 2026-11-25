use bevy::prelude::*;
use bevy_brain::*;

const SCENE_RADIUS: f32 = 3.0;

fn main() {
    println!("Hello, world!");
}

#[derive(Component)]
struct FighterSensors {
    hp: f32,
    dist_to_enemy: f32,
    weigth_diff: f32,
}

impl BrainInput<3> for FighterSensors {
    fn fill(&self, input: &mut [Normalized; 3]) {
        input[0] = self.hp.to_unorm(100.0).invert().into();
        input[1] = self.dist_to_enemy.to_unorm(SCENE_RADIUS).into();
        input[2] = self.weigth_diff.to_snorm(10.0).into();
    }
}
