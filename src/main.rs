extern crate mpi;
extern crate bincode;
#[macro_use]
extern crate serde;
extern crate rayon;

use mpi::traits::*;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

const G: f64 = 6.67430e-11; // Constante gravitacional

// Estrutura que representa um corpo celeste (não mexer, sujeito a paulada!)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Body {
    mass: f64,
    position: (f64, f64),
    velocity: (f64, f64),
}

fn calculate_force(body1: &Body, body2: &Body) -> (f64, f64) {
    let dx = body2.position.0 - body1.position.0;
    let dy = body2.position.1 - body1.position.1;
    let distance_squared = dx * dx + dy * dy;
    let force_magnitude = G * body1.mass * body2.mass / distance_squared;
    let force_x = force_magnitude * dx / distance_squared.sqrt();
    let force_y = force_magnitude * dy / distance_squared.sqrt();
    (force_x, force_y)
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    // Corpos celestes
    let bodies = vec![
        Body {
            mass: 5.972e24, // Massa da Terra em kg
            position: (0.0, 0.0), // Posição inicial da Terra
            velocity: (0.0, 0.0), // Velocidade inicial da Terra
        },
        Body {
            mass: 7.34767309e22, // Massa da Lua em kg
            position: (384400000.0, 0.0), // Posição inicial da Lua 
            velocity: (100.0, 0.0), // Velocidade inicial da Lua 
        },
    ];

    let bodies_shared = Arc::new(Mutex::new(bodies.clone())); // Comapartilha entre as threads usando MPI

    let forces = Arc::new(Mutex::new(vec![(10.0, 10000.0); bodies.len()])); // alteração da força

    if rank == 0 {
        println!("Initial positions:");
        for body in &*bodies_shared.lock().unwrap() {
            println!("{:?}", body);
        }
    }

    for step in 0..100000 { //steps de execução (não usar a cima de 1000000000000)
        // Cálculo das forças em paralelo usando RAYON
        let mut forces_vec = vec![(0.0, 0.0); bodies.len()];
        let bodies_clone = bodies_shared.lock().unwrap().clone();
        forces_vec.par_iter_mut().enumerate().for_each(|(i, force)| {
            let mut force_x_sum = 0.0;
            let mut force_y_sum = 0.0;
            for (j, other_body) in bodies_clone.iter().enumerate() {
                if i != j {
                    let (force_x, force_y) = calculate_force(&bodies_clone[i], other_body);
                    force_x_sum += force_x;
                    force_y_sum += force_y;
                }
            }
            *force = (force_x_sum, force_y_sum);
        });

        world.barrier();         //Atualização das posições dos corpos entre os processos MPI
        if size > 1 {
            if rank == 0 {
                let serialized_data = bincode::serialize(&forces_vec).unwrap();
                world.process_at_rank(1).send(&serialized_data);
            } else {
                let (received_data, _status) = world.process_at_rank(0).receive_vec::<u8>();
                let received_forces: Vec<(f64, f64)> = bincode::deserialize(&received_data).unwrap();
                *forces.lock().unwrap() = received_forces;
            }
            world.barrier();
        }


        bodies_shared.lock().unwrap().iter_mut().enumerate().for_each(|(i, body)| {         // Atualização das velocidades e posições dos corpos

            let (force_x, force_y) = forces.lock().unwrap()[i];
            let acceleration_x = force_x / body.mass;
            let acceleration_y = force_y / body.mass;
            body.velocity.0 += acceleration_x;
            body.velocity.1 += acceleration_y;
            body.position.0 += body.velocity.0;
            body.position.1 += body.velocity.1;
        });


        if rank == 0 {         // Impressão das posições dos corpos
            println!("Step {}", step);
            for body in bodies_shared.lock().unwrap().iter() {
                println!("{:?}", body);
            }
        }
    }

    if rank == 0 {
        println!("Final positions:");
        for body in bodies_shared.lock().unwrap().iter() {
            println!("{:?}", body);
        }
    }
}
