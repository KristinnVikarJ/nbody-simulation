mod bvh_tree;

use bvh_tree::BVHTree;
use flume::{Receiver, Sender};
use pathfinder_geometry::vector::{vec2f, Vector2F as Vec2};
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, IntoParallelIterator};
use std::borrow::Borrow;
use std::f32::consts::TAU;
use std::thread::spawn;
use std::time::Instant;
use std::{mem, ops};
use zstd::bulk::compress;

use error_iter::ErrorIter as _;
#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;
use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use winit::dpi::LogicalSize;
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const HEIGHT: u32 = 100_000;
const RENDER_HEIGHT: u32 = 1250;
const PARTICLE_COUNT: usize = 10_000;
const STEP_SIZE: f32 = 0.1; // Multiplier of current step size, Lower = higher quality
const THETA: f32 = 50.0; // Represents ratio of width/distance, Lower = higher quality

struct World {
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
    build_bvh: f64,
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
            build_bvh: 0.0,
            post_calculations: 0.0,
            sum_gravity: 0.0,
        };
        loop {
            //let before: Vec<Vec2> = world.particles.clone().into_iter().filter(|bleh| within_bounds(&bleh.position)).map(|bleh| bleh.position).collect();
            world.update(STEP_SIZE, &mut counter);
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

#[derive(Clone, Debug)]
pub struct SmallParticle {
    position: Vec2,
    //weight: u32,
}

impl From<Particle> for SmallParticle {
    fn from(val: Particle) -> SmallParticle {
        SmallParticle {
            position: val.position,
            //weight: val.weight,
        }
    }
}
impl From<&Particle> for SmallParticle {
    fn from(val: &Particle) -> SmallParticle {
        SmallParticle {
            position: val.position,
            //weight: val.weight,
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

fn rand_disc() -> Vec2 {
    let theta = fastrand::f32() * TAU;
    Vec2::new(theta.cos(), theta.sin()) * fastrand::f32()
}

fn rand_body(offset: Vec2) -> Particle {
    let pos = rand_disc();
    let vel = rand_disc();

    Particle {
        position: (pos * 25000.0) + offset,
        velocity: vel,
        weight: 1,
    }
}

fn rotate_right(vec: &Vec2) -> Vec2 {
    vec2f(vec.y(), -vec.x())
}

impl World {
    fn new() -> Self {
        let mut particles = Vec::with_capacity(PARTICLE_COUNT);
        let mut rng = rand::thread_rng();
        let circle1 = vec2f(35000.0, 35000.0);
        let circle2 = vec2f(60000.0, 60000.0);
        let c1lenr2 = 15000000.0;
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
        /*
        let sample = Uniform::new(0f32, HEIGHT as f32);

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
        /**/
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
        let offset = vec2f(50000.0, 50000.0);
        for _ in 0..100_000 {
            particles.push(rand_body(offset));
            /*
            particles.push(Particle {
                position: vec2f(rng.sample(sample), rng.sample(sample)),
                velocity: vec2f(0.0, 0.0),
                weight: 1,
            });*/
        }
        println!("len: {}", particles.len());

        Self { particles }
    }

    #[inline(always)]
    fn bvh_sum_gravity(particle: &SmallParticle, tree: &BVHTree, accel: &mut Vec2) {
        match tree {
            BVHTree::Leaf {
                boundary: _,
                children,
            } => {
                for point in children.iter() {
                    calculate_gravity(
                        &particle.position,
                        &point.position,
                        accel,
                        point.weight as f32,
                    );
                }
            }
            BVHTree::Root {
                center_of_gravity,
                boundary,
                total_mass,
                children,
            } => {
                if !boundary.contains(&particle.position)
                    && {let tmp = boundary.size.max(boundary.size.yx()); tmp.x()*tmp.y()}
                        < dist2(&particle.position, &center_of_gravity) * THETA * THETA
                {
                    calculate_gravity(
                        &particle.position,
                        &center_of_gravity,
                        accel,
                        *total_mass as f32,
                    );
                } else {
                    Self::bvh_sum_gravity(particle, &children[0], accel);
                    Self::bvh_sum_gravity(particle, &children[1], accel);
                }
            }
        }
    }

    fn update(&mut self, delta: f32, counter: &mut Counting) {
        let mut timer = Instant::now();
        /*let mut small_particles = self
            .particles
            .iter()
            .map(|part| SmallParticle {
                position: part.position.clone(),
                weight: part.weight,
            })
            .collect::<Vec<SmallParticle>>();*/
        let cloned = self.particles.clone();
        
        let mut bvh_test = BVHTree::from(self.particles.as_mut());
        bvh_test.calculate_gravity();
        counter.build_bvh += timer.elapsed().as_secs_f64();

        timer = Instant::now();

        let acceleration: Vec<Vec2> = cloned
            .par_iter()
            .with_min_len(5000)
            .map(|particle| {
                let mut accel = vec2f(0.0, 0.0);

                Self::bvh_sum_gravity(&particle.clone().into(), &bvh_test, &mut accel);

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
