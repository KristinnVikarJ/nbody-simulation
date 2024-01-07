mod quad_tree;

use flume::{Receiver, Sender};
use pathfinder_geometry::vector::{vec2f, Vector2F as Vec2};
use quad_tree::{QuadTree, QuadTreeType};
use rand::distributions::Uniform;
use rayon::iter::IndexedParallelIterator;
use std::thread::spawn;
use std::time::Instant;
use std::{mem, ops};
use zstd::bulk::compress;

use error_iter::ErrorIter as _;
#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;
use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use rand::Rng;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use winit::dpi::LogicalSize;
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;
use crate::quad_tree::Rectangle;

const HEIGHT: u32 = 100_000;
const RENDER_HEIGHT: u32 = 1250;
const PARTICLE_COUNT: usize = 10_000;
const STEP_SIZE: f32 = 0.05; // Multiplier of current step size, Lower = higher quality
const THETA: f32 = 2.0; // Represents ratio of width/distance, Lower = higher quality

struct World {
    particle_tree: QuadTree,
    particles: Vec<Particle>,
}

fn draw(particles: &Vec<Particle>, frame: &mut [u8]) {
    // Zero out the pixel buffer
    frame.iter_mut().for_each(|m| *m = 0);
    //self.draw_tree(&self.particle_tree, frame);
    //self.draw_weights(&self.particle_tree, frame);
    for particle in particles {
        if !within_bounds(&particle.position) {
            continue;
        }

        let offset = (((particle.position.y() as u32 / (HEIGHT / RENDER_HEIGHT)) * RENDER_HEIGHT)
            + (particle.position.x() as u32 / (HEIGHT / RENDER_HEIGHT)))
            as usize
            * 4;
        if particle.weight > 10 {
            frame[offset] = 0x00; // R
            frame[offset + 1] = 0xff; // G
            frame[offset + 2] = 0x00; // B
            frame[offset + 3] = 0xff;
        } else if frame[offset + 3] != 0xff {
            let velocity = 0x10
                + (((particle.velocity.x().abs() + particle.velocity.y().abs()) * 10.0) as u8)
                    .min(0xef);
            frame[offset] = 0xff; // R
            frame[offset + 1] = 0xff - velocity; // G
            frame[offset + 2] = 0xff - velocity; // B
            if frame[offset + 3] <= 240 {
                frame[offset + 3] += 10; // A
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Counting {
    build_tree: f64,
    calculate_gravity: f64, // part of build_tree
    sum_gravity: f64,
    post_calculations: f64,
}

fn main() -> Result<(), Error> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(RENDER_HEIGHT as f64, RENDER_HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Barnes-Hut Simulation")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(RENDER_HEIGHT, RENDER_HEIGHT, surface_texture)?
    };

    let (tx, rx): (
        Sender<(Vec<Particle>, u64, Counting)>,
        Receiver<(Vec<Particle>, u64, Counting)>,
    ) = flume::bounded(2);

    let mut frames = 0;
    let mut last_updates = 0;
    let mut frame_timer = Instant::now();

    spawn(move || {
        let mut world = World::new();
        let mut updates: u64 = 0;
        let mut counter = Counting {
            build_tree: 0.0,
            calculate_gravity: 0.0,
            post_calculations: 0.0,
            sum_gravity: 0.0,
        };
        loop {
            //let before: Vec<Vec2> = world.particles.clone().into_iter().filter(|bleh| within_bounds(&bleh.position)).map(|bleh| bleh.position).collect();
            world.update(STEP_SIZE, &mut counter, updates % 20 == 0);
            //let diff: Vec<Vec2> = world.particles.iter().enumerate().map(|(i, v)| before[i].sub(&v.position)).collect();

            updates += 1;

            /*if updates % 10 == 0 {
                // Save
                let data;
                unsafe {
                    let transmuted: &[u8] = mem::transmute(diff.as_slice());
                    println!("raw: {}", transmuted.len());
                    data = compress(transmuted, 22).unwrap();
                }
                println!("comp: {}", data.len());
            }*/

            // Send particles over thread
            if !tx.is_full() {
                let _ = tx.try_send((world.particles.clone(), updates, counter.clone()));
            }
        }
    });

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::RedrawRequested(_) => {
                let (particles, updates, counter) = rx.recv().unwrap();
                draw(&particles, pixels.frame_mut());
                frames += 1;
                if frame_timer.elapsed().as_secs() >= 1 {
                    println!("fps: {}\nups: {}", frames, updates - last_updates);
                    println!("step: {}", updates);
                    println!("{:?}", counter);
                    frames = 0;
                    last_updates = updates;
                    frame_timer = Instant::now();
                }
                if let Err(err) = pixels.render() {
                    log_error("pixels.render", err);
                    *control_flow = ControlFlow::Exit;
                    return;
                }
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => (),
        }
        if input.update(&event) {
            // Close events
            if input.key_pressed(VirtualKeyCode::Escape) || input.close_requested() {
                *control_flow = ControlFlow::Exit;
                return;
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                if let Err(err) = pixels.resize_surface(size.width, size.height) {
                    log_error("pixels.resize_surface", err);
                    *control_flow = ControlFlow::Exit;
                }
            }
        }
    });
}

fn log_error<E: std::error::Error + 'static>(method_name: &str, err: E) {
    error!("{method_name}() failed: {err}");
    for source in err.sources().skip(1) {
        error!("  Caused by: {source}");
    }
}

#[derive(Clone, Debug)]
pub struct Particle {
    position: Vec2,
    velocity: Vec2,
    weight: u32,
}

pub struct SmallParticle {
    position: Vec2,
    weight: u32,
}

impl From<Particle> for SmallParticle {
    fn from(val: Particle) -> SmallParticle {
        SmallParticle {
            position: val.position,
            weight: val.weight,
        }
    }
}
impl From<&Particle> for SmallParticle {
    fn from(val: &Particle) -> SmallParticle {
        SmallParticle {
            position: val.position,
            weight: val.weight,
        }
    }
}

#[inline(always)]
fn within_bounds(pos: &Vec2) -> bool {
    pos.y() < HEIGHT as f32 && pos.x() < HEIGHT as f32 && pos.y() >= 0f32 && pos.x() >= 0f32
}

#[inline(always)]
fn dist2(pos1: &Vec2, pos2: &Vec2) -> f32 {
    let diff = *pos1 - *pos2;
    diff.square_length()
}

#[inline(always)]
fn calculate_gravity(particle1: &Vec2, particle2: &Vec2, accel: &mut Vec2, force: f32) {
    let diff = *particle2 - *particle1;

    let sum = diff.x().abs() + diff.y().abs();

    // Validate that we don't have any inf's or NaN etc
    if !sum.is_normal() {
        return;
    }

    let mut distance = diff.square_length();
    // Clamp value
    if distance < 0.001f32 {
        distance = 0.001f32;
    }

    // According to wolfram alpha
    *accel += (diff * force) / (sum * distance);
}

fn make_plummer(n: u32, seed: u32, center_mass: u32) -> Vec<Particle> {
    let total_mass = n + center_mass;
    let mut particles = Vec::with_capacity(n as usize);
    for i in 0..n {
        let mass = total_mass / n;
    }
    particles
}

fn rotate_right(vec: &Vec2) -> Vec2 {
    vec2f(vec.y(), -vec.x())
}

impl World {
    fn new() -> Self {
        let mut particles = Vec::with_capacity(PARTICLE_COUNT);
        let mut rng = rand::thread_rng();
        let sample = Uniform::new(0f32, HEIGHT as f32);
        let circle1 = vec2f(35000.0, 35000.0);
        let circle2 = vec2f(60000.0, 60000.0);
        particles.push(Particle {
            position: circle1.clone(),
            velocity: vec2f(200.0, 250.0),
            weight: 75_000_000,
        });
        particles.push(Particle {
            position: circle2.clone(),
            velocity: vec2f(0.0, 0.0),
            weight: 750_000,
        });

        let c1lenr2 = 15000000.0;
        /*
        for x in 0..((HEIGHT / 14) - 1) {
            for y in 0..((HEIGHT / 14) - 1) {
                let pos = Vec2 {
                    x: x as f64 * 14.0,
                    y: y as f64 * 14.0,
                };
                if dist2(&pos, &circle1) < c1lenr2
                    && dist2(&pos, &circle1) > 500000.0
                    && rng.gen_range(0f64..(c1lenr2 - dist2(&pos, &circle1)) + 1.0) > 6000000.0
                {
                    let velocity = rotate_right(&pos.sub(&circle1))
                        .mul(((750000.0f64).sqrt() / (dist2(&pos, &circle1))).sqrt());
                    particles.push(Particle {
                        position: pos,
                        velocity,
                        weight: 1.0,
                    });
                }
            }
        }*/

        for x in 0..((HEIGHT / 14) - 1) {
            for y in 0..((HEIGHT / 14) - 1) {
                let pos = vec2f(x as f32 * 14.0, y as f32 * 14.0);
                if dist2(&pos, &circle2) < c1lenr2
                    && dist2(&pos, &circle2) > 500000.0
                    && rng.gen_range(0f32..(c1lenr2 - dist2(&pos, &circle2)) + 1.0) > 6000000.0
                {
                    let velocity = rotate_right(&(pos - circle2))
                        * ((750000.0f32).sqrt() / (dist2(&pos, &circle2))).sqrt();
                    particles.push(Particle {
                        position: pos,
                        velocity,
                        weight: 1,
                    });
                }
            }
        }
        for _ in 0..1_000_000 {
            particles.push(Particle {
                position: vec2f(rng.sample(sample), rng.sample(sample)),
                velocity: vec2f(0.0, 0.0),
                weight: 1,
            });
        }
        println!("len: {}", particles.len());

        let particle_tree = QuadTree::new(Rectangle::new(vec2f(0.0, 0.0), HEIGHT as f32));

        Self {
            particle_tree,
            particles,
        }
    }

    // Rebuild parameter temporary
    fn rebuild_tree(&mut self, counter: &mut Counting, rebuild: bool) {
        if rebuild {
            let new_particle_tree = QuadTree::new(Rectangle::new(vec2f(0.0, 0.0), HEIGHT as f32));
            self.particle_tree = new_particle_tree;
        } else {
            self.particle_tree.empty();
        }

        //println!("tree size: {}", size);
        self.particles
            .retain(|particle| within_bounds(&particle.position));

        for particle in self.particles.iter() {
            self.particle_tree.insert(particle.into());
            //new_particle_tree.insert(particle.clone().into())
        }
        //let tpruned = new_particle_tree.prune();
        //println!("expected tree size: {}", new_particle_tree.empty());

        let timer = Instant::now();

        self.particle_tree.calculate_gravity();

        // Get rid of empty nodes
        if !rebuild {
            self.particle_tree.prune();
        }
        //let new_size = self.test_recurse(&self.particle_tree);

        //println!("pruned: {}", pruned);
        //println!("tpruned: {}", tpruned);
        //println!("new_size: {}", new_size);

        counter.calculate_gravity += timer.elapsed().as_secs_f64();

        //self.particle_tree = particle_tree;
    }

    #[inline(always)]
    fn sum_gravity(particle: &SmallParticle, tree: &QuadTree, accel: &mut Vec2) {
        match &tree.tree_type {
            QuadTreeType::Leaf { count: _, children } => {
                for point in children.iter().flatten() {
                    calculate_gravity(
                        &particle.position,
                        &point.position,
                        accel,
                        point.weight as f32,
                    );
                }
            }
            QuadTreeType::Root {
                flags: _,
                total_mass,
                children,
            } => {
                if *total_mass == 0 {
                    return;
                }
                if !tree.boundary.contains(&particle.position)
                    && tree.boundary.height2
                        < dist2(&particle.position, &tree.center_of_gravity) * THETA * THETA
                {
                    calculate_gravity(
                        &particle.position,
                        &tree.center_of_gravity,
                        accel,
                        *total_mass as f32,
                    );
                } else {
                    for child in children.iter().flatten() {
                        Self::sum_gravity(particle, child, accel);
                    }
                }
            }
        }
    }

    fn update(&mut self, delta: f32, counter: &mut Counting, rebuild: bool) {
        let mut timer = Instant::now();
        self.rebuild_tree(counter, rebuild);
        counter.build_tree += timer.elapsed().as_secs_f64();
        timer = Instant::now();
        let acceleration: Vec<Vec2> = self
            .particles
            .par_iter()
            .with_min_len(4000)
            .map(|particle| {
                let mut accel = vec2f(0.0, 0.0);
                Self::sum_gravity(&particle.clone().into(), &self.particle_tree, &mut accel);

                accel
            })
            .collect();
        counter.sum_gravity += timer.elapsed().as_secs_f64();
        timer = Instant::now();
        for (idx, particle) in self.particles.iter_mut().enumerate() {
            particle.velocity += acceleration[idx] * delta;
            let velo = particle.velocity * delta;
            particle.position += velo;
        }
        counter.post_calculations += timer.elapsed().as_secs_f64();
    }
}
