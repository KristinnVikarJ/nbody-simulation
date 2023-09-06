mod quad_tree;

use once_cell::sync::OnceCell;
use quad_tree::{QuadTree, QuadTreeType};
use std::borrow::Borrow;
use std::iter::Sum;
use std::ops;
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{sleep, spawn};
use std::time::{Duration, Instant};

use error_iter::ErrorIter as _;
use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use rand::Rng;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use winit::dpi::LogicalSize;
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

use crate::quad_tree::Rectangle;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const PARTICLE_COUNT: usize = 10_000;
const STEP_SIZE: f64 = 0.0005;

static WORLD: OnceCell<RwLock<World>> = OnceCell::new();

struct World {
    particle_tree: QuadTree,
    particles: Vec<Arc<RwLock<Particle>>>,
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
    WORLD
        .set(RwLock::from(world))
        .unwrap_or_else(|_| panic!("world broken"));

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
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

#[allow(dead_code)]
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

    fn div(self: &Vec2, scalar: f64) -> Vec2 {
        Vec2::new_from(self.x / scalar, self.y / scalar)
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

impl<'a> Sum<Self> for Vec2 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self { x: 0.0, y: 0.0 }, |mut a, b| {
            a.x += b.x;
            a.y += b.y;
            a
        })
    }
}

// Assume constant mass of 1 for now
pub struct Particle {
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

fn calculate_gravity(particle1: &Vec2, particle2: &Vec2, force: f64) -> Vec2 {
    let xdiff = particle2.x - particle1.x;
    let ydiff = particle2.y - particle1.y;

    // Validate that we don't have any inf's or NaN etc
    if !(xdiff + ydiff).is_normal() {
        return Vec2::new();
    }

    let sum = ydiff.abs() + xdiff.abs();

    let mut distance = (xdiff * xdiff) + (ydiff * ydiff);
    // Clamp value
    if distance < 0.0001f64 {
        distance = 0.0001f64;
    }

    // According to wolfram
    //let sd = sum * distance;

    let reduced_force = force / distance;

    Vec2 {
        x: (xdiff / sum) * reduced_force,
        y: (ydiff / sum) * reduced_force,
    }
}

impl World {
    fn new() -> Self {
        let mut particles = Vec::with_capacity(PARTICLE_COUNT);
        let mut rng = rand::thread_rng();

        let circle1 = Vec2 { x: 960.0, y: 540.0 };

        let circle2 = Vec2 { x: 960.0, y: 540.0 }; // Teeny-tiny bit off-center, so we dont have particles inside each other

        let c1lenr2 = 6000.0;

        for x in 0..WIDTH - 1 {
            for y in 0..HEIGHT - 1 {
                let pos = Vec2 {
                    x: x as f64,
                    y: y as f64,
                };
                if dist2(&pos, &circle1) < c1lenr2
                //&& rng.gen_range(0f64..(c1lenr2 - dist2(&pos, &circle2)) + 1.0) > 100.0
                {
                    let velocity = rotate_right(&pos.sub(&circle1)).mul(0.4);
                    particles.push(Arc::from(RwLock::from(Particle {
                        position: pos,
                        velocity,
                    })));
                }
            }
        }

        // Inner circle
        for x in 0..WIDTH - 1 {
            for y in 0..HEIGHT - 1 {
                let pos = Vec2 {
                    x: x as f64 + 0.1,
                    y: y as f64 + 0.1,
                };
                if dist2(&pos, &circle2) < 2000.0 {
                    let velocity = rotate_right(&pos.sub(&circle2)).mul(0.35);
                    particles.push(Arc::from(RwLock::from(Particle {
                        position: pos,
                        velocity,
                    })));
                }
            }
        }

        for _ in 0..1_000 {
            particles.push(Arc::from(RwLock::from(Particle {
                position: Vec2 {
                    x: rng.gen_range(0f64..WIDTH as f64),
                    y: rng.gen_range(0f64..HEIGHT as f64),
                },
                velocity: Vec2 {
                    x: rng.gen_range(-0.1..0.1),
                    y: rng.gen_range(-0.1..0.1),
                },
            })));
        }

        println!("len: {}", particles.len());

        let particle_tree = QuadTree::new(Rectangle {
            height: HEIGHT as f64,
            width: WIDTH as f64,
            offset: Vec2::new(),
        });

        Self {
            particle_tree,
            particles,
        }
    }

    fn rebuild_tree(&mut self) {
        let mut particle_tree = QuadTree::new(Rectangle {
            height: HEIGHT as f64,
            width: WIDTH as f64,
            offset: Vec2::new(),
        });

        for p in &self.particles {
            particle_tree.insert(p.clone());
        }

        self.particle_tree = particle_tree;
    }

    fn update(&mut self, delta: f64) {
        self.rebuild_tree();
        let acceleration: Vec<Vec2> = self
            .particles
            .par_iter()
            .map(|locked_particle| {
                let particle = locked_particle.read().unwrap();
                let mut current = &self.particle_tree;
                let mut accel = Vec2::new();
                loop {
                    match &current.tree_type {
                        QuadTreeType::Leaf { points } => {
                            for locked_point in points {
                                if !std::ptr::eq(locked_particle, locked_point) {
                                    let point = locked_point.read().unwrap();
                                    accel +=
                                        calculate_gravity(&particle.position, &point.position, 1.0);
                                }
                            }
                            break;
                        }
                        QuadTreeType::Root {
                            count: _,
                            ne,
                            se,
                            sw,
                            nw,
                        } => {
                            let hori_half =
                                current.boundary.offset.x + (current.boundary.width / 2.0);
                            let vert_half =
                                current.boundary.offset.y + (current.boundary.height / 2.0);

                            let north = particle.position.y <= vert_half;
                            let west = particle.position.x <= hori_half;

                            match north {
                                true => match west {
                                    true => {
                                        current = nw;
                                    }
                                    false => {
                                        current = ne;
                                    }
                                },
                                false => match west {
                                    true => {
                                        current = sw;
                                    }
                                    false => {
                                        current = se;
                                    }
                                },
                            };

                            for tree in [ne, se, sw, nw] {
                                if !std::ptr::eq(current, tree.as_ref()) {
                                    accel += calculate_gravity(
                                        &particle.position,
                                        &tree.center_of_gravity,
                                        tree.total_mass,
                                    );
                                }
                            }
                        }
                    }
                }
                accel
            })
            .collect();

        for (idx, locked_particle) in self.particles.iter_mut().enumerate() {
            let mut particle = locked_particle.write().unwrap();
            particle.velocity += &acceleration[idx] * delta;
            let velo = particle.velocity.mul(delta);
            particle.position += velo;
        }
    }

    fn draw_tree(&self, node: &QuadTree, frame: &mut [u8]) {
        for x in [
            (node.boundary.offset.x as usize),
            ((node.boundary.offset.x + node.boundary.width - 1.0) as usize),
        ] {
            for y in (node.boundary.offset.y as usize)
                ..((node.boundary.offset.y + node.boundary.height) as usize)
            {
                let offset = ((y as u32 * WIDTH) + x as u32) as usize * 4;
                frame[offset] = 0xff; // R
                frame[offset + 1] = 0xff; // G
                frame[offset + 2] = 0xff; // B
                frame[offset + 3] = 0xff; // A
            }
        }
        for x in (node.boundary.offset.x as usize)
            ..((node.boundary.offset.x + node.boundary.width) as usize)
        {
            for y in [
                (node.boundary.offset.y as usize),
                ((node.boundary.offset.y + node.boundary.height - 1.0) as usize),
            ] {
                let offset = ((y as u32 * WIDTH) + x as u32) as usize * 4;
                frame[offset] = 0xff; // R
                frame[offset + 1] = 0xff; // G
                frame[offset + 2] = 0xff; // B
                frame[offset + 3] = 0xff; // A
            }
        }
        match &node.tree_type {
            QuadTreeType::Leaf { points: _ } => {
                // we done here boys
            }
            QuadTreeType::Root {
                count: _,
                ne,
                se,
                sw,
                nw,
            } => {
                self.draw_tree(ne, frame);
                self.draw_tree(se, frame);
                self.draw_tree(sw, frame);
                self.draw_tree(nw, frame);
            }
        }
    }

    /// Draw the `World` state to the frame buffer.
    ///
    /// Assumes the default texture format: `wgpu::TextureFormat::Rgba8UnormSrgb`
    fn draw(&self, frame: &mut [u8]) {
        // Zero out the pixel buffer
        frame.iter_mut().for_each(|m| *m = 0);
        //self.draw_tree(&self.particle_tree, frame);
        for locked_particle in self.particles.iter() {
            let particle = locked_particle.read().unwrap();
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
