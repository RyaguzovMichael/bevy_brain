use bevy::prelude::*;
use bevy_brain::prelude::*;
use bevy_brain::scripted_brain::ScriptedBrain;

const SCENE_RADIUS: f32 = 3.0;
const WEAPON_RANGE: f32 = 0.5;

fn main() {
    let sensors = FighterSensors {
        hp: 30.0,
        dist_to_enemy: 2.0,
        weigth_diff: 60.0,
    };
    let brain = ScriptedBrain::create(ai_logic);
    let output = brain.think(&sensors);

    println!("{:?}", output);
}

fn ai_logic(input: &FighterSensors) -> FighterActions {
    if input.dist_to_enemy > WEAPON_RANGE {
        return FighterActions { attack: false };
    }
    return FighterActions {
        attack: input.hp >= 60.0,
    };
}

#[derive(Component)]
struct FighterSensors {
    hp: f32,
    dist_to_enemy: f32,
    weigth_diff: f32,
}

#[derive(Debug, Component, Default)]
struct FighterActions {
    attack: bool,
}

impl BrainInput<3> for FighterSensors {}
impl BrainOutput<1> for FighterActions {}
