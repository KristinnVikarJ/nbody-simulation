use std::sync::{Arc, RwLock};

use crate::{Particle, Vec2};

#[derive(Clone)]
pub struct Rectangle {
    pub offset: Vec2,
    pub height: f64,
    pub width: f64,
}

impl Rectangle {
    pub fn new(offset: Vec2, width: f64, height: f64) -> Self {
        Self {
            offset,
            height,
            width,
        }
    }

    pub fn contains(self: &Rectangle, other: &Vec2) -> bool {
        other.y > self.offset.y
            && other.x > self.offset.x
            && other.x < (self.offset.x + self.width)
            && other.y < (self.offset.y + self.height)
    }

    pub fn offset(self: &Rectangle, x: f64, y: f64) -> Vec2 {
        Vec2::new_from(self.offset.x + x, self.offset.y + y)
    }
}

pub struct QuadTree {
    pub boundary: Rectangle,
    pub center_of_gravity: Vec2,
    pub total_mass: f64,
    pub tree_type: QuadTreeType,
}

pub enum QuadTreeType {
    Leaf {
        points: Vec<Arc<RwLock<Particle>>>,
    },
    Root {
        count: usize,
        ne: Box<QuadTree>,
        se: Box<QuadTree>,
        sw: Box<QuadTree>,
        nw: Box<QuadTree>,
    },
}

impl QuadTree {
    const MAX_CAPACITY: usize = 4;
    pub fn new(boundary: Rectangle) -> Self {
        QuadTree {
            boundary,
            tree_type: QuadTreeType::Leaf { points: Vec::new() },
            center_of_gravity: Vec2::new(),
            total_mass: 0.0,
        }
    }

    pub fn count(&self) -> usize {
        match self.tree_type {
            QuadTreeType::Leaf { ref points } => return points.len(),
            QuadTreeType::Root {
                ne: _,
                se: _,
                sw: _,
                nw: _,
                count,
            } => return count,
        }
    }

    pub fn insert(&mut self, point: Arc<RwLock<Particle>>) {
        match self.tree_type {
            QuadTreeType::Leaf { ref mut points } => {
                if points.len() == QuadTree::MAX_CAPACITY {
                    self.subdivide();
                    self.insert(point);
                } else {
                    points.push(point.clone());
                }
                match self.tree_type {
                    QuadTreeType::Leaf {
                        points: ref inner_points,
                    } => {
                        // TODO: make both account for mass
                        // TODO: Optimization? we're calling particle.read() on all inserts
                        // Maybe we should be doing this math once in the end instead of repeatedly
                        let sum_vec: Vec2 = inner_points
                            .iter()
                            .map(|particle| {
                                let part = particle.read().unwrap();
                                part.position.clone()
                            })
                            .sum();
                        self.total_mass += 1.0;
                        self.center_of_gravity = sum_vec.div(self.total_mass);
                    }
                    QuadTreeType::Root {
                        count: _,
                        ref ne,
                        ref se,
                        ref sw,
                        ref nw,
                    } => {
                        let total_mass =
                            ne.total_mass + se.total_mass + sw.total_mass + nw.total_mass;

                        let big_thing = ne.center_of_gravity.mul(ne.total_mass)
                            + se.center_of_gravity.mul(se.total_mass)
                            + sw.center_of_gravity.mul(se.total_mass)
                            + nw.center_of_gravity.mul(nw.total_mass);

                        self.center_of_gravity = big_thing.div(total_mass);
                        self.total_mass = total_mass;
                    }
                }
            }
            QuadTreeType::Root {
                ref mut ne,
                ref mut se,
                ref mut sw,
                ref mut nw,
                ref mut count,
            } => {
                let hori_half = self.boundary.offset.x + (self.boundary.width / 2.0);
                let vert_half = self.boundary.offset.y + (self.boundary.height / 2.0);
                let north;
                let west;
                {
                    let p = point.clone();
                    let punlocked = p.read().unwrap();
                    north = punlocked.position.y <= vert_half;
                    west = punlocked.position.x <= hori_half;
                }

                match north {
                    true => match west {
                        true => nw.insert(point),
                        false => ne.insert(point),
                    },
                    false => match west {
                        true => sw.insert(point),
                        false => se.insert(point),
                    },
                };

                *count += 1;
            }
        }
    }

    fn subdivide(&mut self) {
        match self.tree_type {
            QuadTreeType::Leaf { ref mut points } => {
                let new_width = self.boundary.width / 2.0;
                let new_height = self.boundary.height / 2.0;

                let mut new = QuadTree {
                    boundary: self.boundary.clone(),
                    tree_type: QuadTreeType::Root {
                        ne: Box::new(QuadTree::new(Rectangle::new(
                            self.boundary.offset(new_width, 0.0),
                            new_width,
                            new_height,
                        ))),
                        se: Box::new(QuadTree::new(Rectangle::new(
                            self.boundary.offset(new_width, new_height),
                            new_width,
                            new_height,
                        ))),
                        sw: Box::new(QuadTree::new(Rectangle::new(
                            self.boundary.offset(0.0, new_height),
                            new_width,
                            new_height,
                        ))),
                        nw: Box::new(QuadTree::new(Rectangle::new(
                            self.boundary.offset.clone(),
                            new_width,
                            new_height,
                        ))),
                        count: points.len(),
                    },
                    center_of_gravity: self.center_of_gravity.clone(),
                    total_mass: self.total_mass,
                };
                for p in points {
                    new.insert(p.clone());
                }
                *self = new;
            }
            _ => {}
        }
    }
}
