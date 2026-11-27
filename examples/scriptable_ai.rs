use bevy::prelude::*;
use bevy_brain::prelude::*;
use burn::tensor::{Device, Tensor};

const SCENE_RADIUS: f32 = 3.0;

fn main() {
    let sensors = FighterSensors {
        hp: 30.0,
        dist_to_enemy: 2.0,
        weigth_diff: 60.0,
    };
    let device = Device::flex();
    let mut input_arr: [Normalized; 3] = Default::default();
    sensors.fill(&mut input_arr);
    let input_data: [[f32; 1]; 3] = [
        [input_arr[0].get()],
        [input_arr[1].get()],
        [input_arr[2].get()],
    ];
    let input_tensor: Tensor<2> = Tensor::from_data(input_data, &device);

    let hidden_layer: Tensor<2> = Tensor::from_data([[2.0, 3.0, -1.0], [4.0, 5.0, -2.0]], &device);
    let biases: Tensor<2> = Tensor::from_data([[-1.0],[2.0]], &device);

    let output_tensor = hidden_layer.matmul(input_tensor) + biases;

    println!("{}", output_tensor);
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
