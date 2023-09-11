mod quad_tree;

use flume::{Receiver, Sender};
use quad_tree::{QuadTree, QuadTreeType};
use std::borrow::Borrow;
use std::iter::Sum;
use std::ops;
use std::thread::spawn;
use std::time::Instant;

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

const HEIGHT: u32 = 1000;
const PARTICLE_COUNT: usize = 10_000;
const STEP_SIZE: f64 = 0.001; // Multiplier of current step size, Lower = higher quality
const THETA: f64 = 0.9; // Represents ratio of width/distance, Lower = higher quality

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

        let offset =
            ((particle.position.y as u32 * HEIGHT) + particle.position.x as u32) as usize * 4;
        let velocity = 0x10
            + (((particle.velocity.x.abs() + particle.velocity.y.abs()) * 10.0) as u8).min(0xef);
        frame[offset] = 0xff; // R
        frame[offset + 1] = 0xff - velocity; // G
        frame[offset + 2] = 0xff - velocity; // B
        if frame[offset + 3] < 0xff && frame[offset + 3] + 10 > frame[offset + 3] {
            frame[offset + 3] += 10; // A
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
        let size = LogicalSize::new(HEIGHT as f64, HEIGHT as f64);
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
        Pixels::new(HEIGHT, HEIGHT, surface_texture)?
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
            let particles = world.update(STEP_SIZE, &mut counter);
            updates += 1;

            // Send particles over thread
            let _ = tx.try_send((particles, updates, counter.clone()));
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
#[derive(Clone)]
pub struct Particle {
    position: Vec2,
    velocity: Vec2,
}

#[inline(always)]
fn within_bounds(pos: &Vec2) -> bool {
    pos.y < HEIGHT as f64 && pos.x < HEIGHT as f64 && pos.y >= 0f64 && pos.x >= 0f64
}

#[inline(always)]
fn dist2(pos1: &Vec2, pos2: &Vec2) -> f64 {
    let xdiff = pos2.x - pos1.x;
    let ydiff = pos2.y - pos1.y;
    (xdiff * xdiff) + (ydiff * ydiff)
}

#[inline(always)]
fn dist(pos1: &Vec2, pos2: &Vec2) -> f64 {
    dist2(pos1, pos2).sqrt()
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
    if distance < 0.001f64 {
        distance = 0.001f64;
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
        /*
        let circle1 = Vec2 { x: 500.0, y: 500.0 };

        let c1lenr2 = 10000.0;

        for x in 0..HEIGHT - 1 {
            for y in 0..HEIGHT - 1 {
                let pos = Vec2 {
                    x: x as f64,
                    y: y as f64,
                };
                if dist2(&pos, &circle1) < c1lenr2
                    && rng.gen_range(0f64..(c1lenr2 - dist2(&pos, &circle1)) + 1.0) > 1500.0
                {
                    let velocity = rotate_right(&pos.sub(&circle1)).mul(0.175);
                    particles.push(Particle {
                        position: pos,
                        velocity,
                    });
                }
            }
        }
        */
        for _ in 0..25_000 {
            particles.push(Particle {
                position: Vec2 {
                    x: rng.gen_range(0f64..HEIGHT as f64),
                    y: rng.gen_range(0f64..HEIGHT as f64),
                },
                velocity: Vec2::new(),
            });
        }
        println!("len: {}", particles.len());

        let particle_tree = QuadTree::new(Rectangle::new(Vec2::new(), HEIGHT as f64));

        Self {
            particle_tree,
            particles,
        }
    }

    fn rebuild_tree(&mut self, counter: &mut Counting) {
        let mut particle_tree = QuadTree::new(Rectangle::new(Vec2::new(), HEIGHT as f64));

        let mut offset = 0;
        for (idx, particle) in self.particles.clone().iter().enumerate() {
            if particle_tree.boundary.contains(&particle.position) {
                particle_tree.insert(particle.clone());
            } else {
                self.particles.remove(idx - offset);
                offset += 1;
            }
        }

        let timer = Instant::now();

        particle_tree.calculate_gravity();

        counter.calculate_gravity += timer.elapsed().as_secs_f64();

        self.particle_tree = particle_tree;
    }

    fn sum_gravity(particle: &Particle, tree: &QuadTree, accel: &mut Vec2) {
        match &tree.tree_type {
            QuadTreeType::Leaf { points } => {
                for point in points {
                    if !std::ptr::eq(particle, point) {
                        *accel += calculate_gravity(&particle.position, &point.position, 1.0);
                    }
                }
            }
            QuadTreeType::Root {
                total_mass,
                children
            } => {
                if !tree.boundary.contains(&particle.position)
                    && tree.boundary.height2 / dist2(&particle.position, &tree.center_of_gravity)
                        < THETA
                {
                    *accel +=
                        calculate_gravity(&particle.position, &tree.center_of_gravity, *total_mass);
                } else {
                    for child in children {
                        Self::sum_gravity(particle, child, accel);
                    }
                }
            }
        }
    }

    fn update(&mut self, delta: f64, counter: &mut Counting) -> Vec<Particle> {
        let mut timer = Instant::now();
        self.rebuild_tree(counter);
        counter.build_tree += timer.elapsed().as_secs_f64();
        timer = Instant::now();
        let acceleration: Vec<Vec2> = self
            .particles
            .par_iter()
            .map(|particle| {
                let mut accel = Vec2::new();
                Self::sum_gravity(particle, &self.particle_tree, &mut accel);

                accel
            })
            .collect();
        counter.sum_gravity += timer.elapsed().as_secs_f64();
        timer = Instant::now();
        for (idx, particle) in self.particles.iter_mut().enumerate() {
            particle.velocity += &acceleration[idx] * delta;
            let velo = particle.velocity.mul(delta);
            particle.position += velo;
        }
        counter.post_calculations += timer.elapsed().as_secs_f64();

        self.particles.clone()
    }

    fn draw_tree(&self, node: &QuadTree, frame: &mut [u8]) {
        if node.get_total_mass() > 3000.0 {
            for x in [
                (node.boundary.offset.x as usize),
                ((node.boundary.offset.x + node.boundary.height - 1.0) as usize),
            ] {
                for y in (node.boundary.offset.y as usize)
                    ..((node.boundary.offset.y + node.boundary.height) as usize)
                {
                    let offset = ((y as u32 * HEIGHT) + x as u32) as usize * 4;
                    frame[offset] = 0xff; // R
                    frame[offset + 1] = 0xff; // G
                    frame[offset + 2] = 0xff; // B
                    frame[offset + 3] = 0xff; // A
                }
            }
            for x in (node.boundary.offset.x as usize)
                ..((node.boundary.offset.x + node.boundary.height) as usize)
            {
                for y in [
                    (node.boundary.offset.y as usize),
                    ((node.boundary.offset.y + node.boundary.height - 1.0) as usize),
                ] {
                    let offset = ((y as u32 * HEIGHT) + x as u32) as usize * 4;
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
                    total_mass: _,
                    children
                } => {
                    for child in children {
                        self.draw_tree(child, frame);
                    }
                }
            }
        }
    }

    fn draw_weights(&self, node: &QuadTree, frame: &mut [u8]) {
        if node.get_total_mass() > 3000.0 {
            let x = node.center_of_gravity.x;
            let y = node.center_of_gravity.y;
            let offset = ((y as u32 * HEIGHT) + x as u32) as usize * 4;
            if offset >= 8294400 {
                return;
            }
            frame[offset] = 0; // R
            frame[offset + 1] = 0xff; // G
            frame[offset + 2] = 0; // B
            frame[offset + 3] = 0xff; // A
            match &node.tree_type {
                QuadTreeType::Leaf { points: _ } => {
                    // we done here boys
                }
                QuadTreeType::Root {
                    total_mass: _,
                    children
                } => {
                    for child in children {
                        self.draw_weights(child, frame);
                    }
                }
            }
        }
    }
}
