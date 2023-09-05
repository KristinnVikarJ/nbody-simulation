//mod quad_tree;

use once_cell::sync::OnceCell;
use std::borrow::Borrow;
use std::ops;
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{sleep, spawn};
use std::time::{Duration, Instant};

use error_iter::ErrorIter as _;
use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use rand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use winit::dpi::LogicalSize;
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const PARTICLE_COUNT: usize = 10_000;
const STEP_SIZE: f64 = 0.01; //0.001;

static WORLD: OnceCell<RwLock<World>> = OnceCell::new();

#[derive(Debug)]
struct World {
    particles: Vec<Particle>,
}

fn main() -> Result<(), Error> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Hello Pixels")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture)?
    };
    let world = World::new();
    WORLD.set(RwLock::from(world)).unwrap();

    let mut frames = 0;
    let updates = Arc::new(Mutex::from(0));
    let mut frame_timer = Instant::now();

    // Clone the arc
    let updates2 = updates.clone();
    spawn(move || loop {
        WORLD
            .get()
            .expect("first")
            .write()
            .expect("second")
            .update(STEP_SIZE);
        let mut updates_data = updates2.lock().unwrap();
        *updates_data += 1;
        sleep(Duration::from_nanos(500)); // Max 2000 updates/s
    });

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::RedrawRequested(_) => {
                WORLD
                    .get()
                    .unwrap()
                    .read()
                    .unwrap()
                    .draw(pixels.frame_mut());
                frames += 1;
                if frame_timer.elapsed().as_secs() >= 1 {
                    println!("fps: {}", frames);
                    let mut updates_data = updates.lock().unwrap();
                    println!("ups: {}", updates_data);
                    frames = 0;
                    *updates_data = 0;
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
struct Vec2 {
    x: f64,
    y: f64,
}

impl Vec2 {
    fn new() -> Self {
        Self { x: 0f64, y: 0f64 }
    }

    fn new_from(x: f64, y: f64) -> Self {
        Vec2 { x, y }
    }

    fn sub(self: &Vec2, other: &Vec2) -> Vec2 {
        Vec2::new_from(self.x - other.x, self.y - other.y)
    }

    fn add(self: &Vec2, other: &Vec2) -> Vec2 {
        Vec2::new_from(self.x + other.x, self.y + other.y)
    }

    fn mul(self: &Vec2, scalar: f64) -> Vec2 {
        Vec2::new_from(self.x * scalar, self.y * scalar)
    }

    fn dot(self: &Vec2, other: &Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    fn length(self: &Vec2) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    fn normalize(self: &Vec2) -> Vec2 {
        let length = self.length();
        // Roughly zero
        if length <= 0.0001 {
            return Vec2::new();
        }
        Vec2::new_from(self.x / length, self.y / length)
    }
}

fn rotate_right(vec: &Vec2) -> Vec2 {
    Vec2::new_from(vec.y, -vec.x)
}

impl<R: Borrow<Vec2>> ops::AddAssign<R> for Vec2 {
    fn add_assign(&mut self, other: R) {
        let other = other.borrow();
        self.x += other.x;
        self.y += other.y;
    }
}

impl<R> ops::Add<R> for Vec2
where
    Vec2: ops::AddAssign<R>,
{
    type Output = Vec2;
    fn add(mut self, other: R) -> Self::Output {
        self += other;
        self
    }
}

impl<'a, R> ops::Add<R> for &'a Vec2
where
    Vec2: ops::Add<R>,
{
    type Output = <Vec2 as ops::Add<R>>::Output;
    fn add(self, other: R) -> Self::Output {
        self.clone() + other
    }
}

impl ops::Mul<f64> for &Vec2 {
    type Output = Vec2;
    fn mul(self, rhs: f64) -> Self::Output {
        Vec2 {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

// Assume constant mass of 1 for now
#[derive(Debug)]
struct Particle {
    position: Vec2,
    velocity: Vec2,
}

#[inline(always)]
fn within_bounds(pos: &Vec2) -> bool {
    pos.y < HEIGHT as f64 && pos.x < WIDTH as f64 && pos.y >= 0f64 && pos.x >= 0f64
}

#[inline(always)]
fn dist2(pos1: &Vec2, pos2: &Vec2) -> f64 {
    let xdiff = pos2.x - pos1.x;
    let ydiff = pos2.y - pos1.y;
    (xdiff * xdiff) + (ydiff * ydiff)
}

fn calculate_gravity(particle1: &Particle, particle2: &Particle) -> Vec2 {
    let xdiff = particle2.position.x - particle1.position.x;
    let ydiff = particle2.position.y - particle1.position.y;

    // Validate that we don't have any inf's or NaN etc
    if !(xdiff + ydiff).is_normal() {
        return Vec2::new();
    }

    let sum = ydiff.abs() + xdiff.abs();

    let mut distance = (xdiff * xdiff) + (ydiff * ydiff);
    // Clamp value
    if distance < 0.001f64 {
        distance = 0.001f64;
    }

    // According to wolfram
    let sd = sum * distance;

    // Since we assume mass to be 1 for all particles
    Vec2 {
        x: xdiff / sd,
        y: ydiff / sd,
    }

    /*
    let force = 1f64/distance;

    Vec2 { x: (xdiff/sum)*force, y: (ydiff/sum)*force }
    */
}

impl World {
    /// Create a new `World` instance that can draw a moving box.
    fn new() -> Self {
        let mut particles = Vec::with_capacity(PARTICLE_COUNT);
        let mut rng = rand::thread_rng();

        let circle1 = Vec2 { x: 320.0, y: 540.0 };
        let circle2 = Vec2 {
            x: 1600.0,
            y: 540.0,
        };

        let c1lenr2 = 6000.0;

        for x in 0..WIDTH - 1 {
            for y in 0..HEIGHT - 1 {
                let pos = Vec2 {
                    x: x as f64,
                    y: y as f64,
                };
                if dist2(&pos, &circle2) < c1lenr2 {
                    if rng.gen_range(0f64..(c1lenr2 - dist2(&pos, &circle2)) + 1.0) > 100.0 {
                        let velocity = rotate_right(&pos.sub(&circle2)).mul(0.4);
                        particles.push(Particle {
                            position: pos,
                            velocity: velocity,
                        });
                    }
                }
            }
        }

        for _ in 0..1_000 {
            particles.push(Particle {
                position: Vec2 {
                    x: rng.gen_range(0f64..WIDTH as f64),
                    y: rng.gen_range(0f64..HEIGHT as f64),
                },
                velocity: Vec2 {
                    x: rng.gen_range(-0.1..0.1),
                    y: rng.gen_range(-0.1..0.1),
                },
            });
        }

        println!("len: {}", particles.len());
        /*
         */
        Self { particles }
    }

    fn update(&mut self, delta: f64) {
        let acceleration: Vec<Vec2> = self
            .particles
            .par_iter()
            .enumerate()
            .map(|(idx, particle)| {
                self.particles.iter().enumerate().fold(
                    Vec2::new(),
                    |acc, (inner_idx, inner_particle)| {
                        if idx == inner_idx {
                            return acc;
                        }
                        let gravity_vector = calculate_gravity(particle, inner_particle);
                        acc + gravity_vector
                    },
                )
            })
            .collect();

        for (idx, particle) in self.particles.iter_mut().enumerate() {
            particle.velocity += &acceleration[idx] * delta;
            particle.position += &particle.velocity * delta;
        }
    }

    /// Draw the `World` state to the frame buffer.
    ///
    /// Assumes the default texture format: `wgpu::TextureFormat::Rgba8UnormSrgb`
    fn draw(&self, frame: &mut [u8]) {
        // Clear the pixel buffer
        for pixel in frame.chunks_exact_mut(4) {
            pixel[0] = 0x00; // R
            pixel[1] = 0x00; // G
            pixel[2] = 0x00; // B
            pixel[3] = 0x00; // A
        }
        for particle in self.particles.iter() {
            if !within_bounds(&particle.position) {
                continue;
            }

            let offset =
                ((particle.position.y as u32 * WIDTH) + particle.position.x as u32) as usize * 4;
            let velocity = 0x10
                + (((particle.velocity.x.abs() + particle.velocity.y.abs()) * 10.0) as u8)
                    .min(0xef);
            frame[offset] = 0xff; // R
            frame[offset + 1] = 0xff - velocity; // G
            frame[offset + 2] = 0xff - velocity; // B
            if frame[offset + 3] < 0xff && frame[offset + 3] + 10 > frame[offset + 3] {
                frame[offset + 3] += 10; // A
            }
        }
    }
}
