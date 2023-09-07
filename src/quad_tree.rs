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
    pub tree_type: QuadTreeType,
}

pub enum QuadTreeType {
    Leaf {
        points: Vec<Arc<RwLock<Particle>>>,
        sum_vec: Vec2,
    },
    Root {
        total_mass: f64,
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
            tree_type: QuadTreeType::Leaf {
                points: Vec::new(),
                sum_vec: Vec2::new(),
            },
            center_of_gravity: Vec2::new(),
        }
    }

    pub fn count(&self) -> usize {
        match self.tree_type {
            QuadTreeType::Leaf {
                ref points,
                sum_vec: _,
            } => return points.len(),
            QuadTreeType::Root {
                ne: _,
                se: _,
                sw: _,
                nw: _,
                total_mass: _,
                count,
            } => return count,
        }
    }

    pub fn get_total_mass(&self) -> f64 {
        match &self.tree_type {
            QuadTreeType::Leaf { points, sum_vec: _ } => points.len() as f64,
            QuadTreeType::Root {
                total_mass,
                count: _,
                ne: _,
                se: _,
                sw: _,
                nw: _,
            } => *total_mass,
        }
    }

    pub fn insert(&mut self, point: Arc<RwLock<Particle>>, pos: &Vec2) {
        match self.tree_type {
            QuadTreeType::Leaf {
                ref mut points,
                ref mut sum_vec,
            } => {
                if points.len() == QuadTree::MAX_CAPACITY {
                    self.subdivide();
                    self.insert(point, pos);
                } else {
                    points.push(point.clone());
                    *sum_vec = sum_vec.add(pos);
                }
            }
            QuadTreeType::Root {
                ref mut ne,
                ref mut se,
                ref mut sw,
                ref mut nw,
                ref mut count,
                total_mass: _,
            } => {
                let hori_half = self.boundary.offset.x + (self.boundary.width / 2.0);
                let vert_half = self.boundary.offset.y + (self.boundary.height / 2.0);
                let north = pos.y <= vert_half;
                let west = pos.x <= hori_half;

                match north {
                    true => match west {
                        true => nw.insert(point, pos),
                        false => ne.insert(point, pos),
                    },
                    false => match west {
                        true => sw.insert(point, pos),
                        false => se.insert(point, pos),
                    },
                };

                *count += 1;
            }
        }
    }

    fn subdivide(&mut self) {
        match self.tree_type {
            QuadTreeType::Leaf {
                ref mut points,
                sum_vec: _,
            } => {
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
                        total_mass: points.len() as f64,
                    },
                    center_of_gravity: self.center_of_gravity.clone(),
                };
                for p in points {
                    new.insert(p.clone(), &p.read().unwrap().position);
                }
                *self = new;
            }
            _ => {}
        }
    }

    pub fn calculate_gravity(&mut self) {
        match self.tree_type {
            QuadTreeType::Leaf {
                ref points,
                ref sum_vec,
            } => {
                let len = points.len();
                if len > 0 {
                    self.center_of_gravity = sum_vec.div(len as f64);
                }
            }
            QuadTreeType::Root {
                count: _,
                ref mut ne,
                ref mut se,
                ref mut sw,
                ref mut nw,
                ref mut total_mass,
            } => {
                ne.calculate_gravity();
                se.calculate_gravity();
                sw.calculate_gravity();
                nw.calculate_gravity();

                let mass = ne.get_total_mass()
                    + se.get_total_mass()
                    + sw.get_total_mass()
                    + nw.get_total_mass();

                let big_thing = ne.center_of_gravity.mul(ne.get_total_mass())
                    + se.center_of_gravity.mul(se.get_total_mass())
                    + sw.center_of_gravity.mul(se.get_total_mass())
                    + nw.center_of_gravity.mul(nw.get_total_mass());

                self.center_of_gravity = big_thing.div(mass);
                *total_mass = mass;
            }
        }
    }
}
