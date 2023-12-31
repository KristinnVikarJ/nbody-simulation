use std::mem;

use crate::{Particle, SmallVec2, Vec2};

#[derive(Clone)]
pub struct Rectangle {
    pub offset: SmallVec2,
    pub height: f32,
    pub height2: f32,
}

impl Rectangle {
    pub fn new(offset: SmallVec2, height: f32) -> Self {
        Self {
            offset,
            height,
            height2: height * height,
        }
    }

    pub fn contains(self: &Rectangle, other: &Vec2) -> bool {
        other.y as f32 > self.offset.y
            && (other.x as f32) > self.offset.x
            && (other.x as f32) < (self.offset.x + self.height)
            && (other.y as f32) < (self.offset.y + self.height)
    }

    pub fn offset(self: &Rectangle, x: f32, y: f32) -> SmallVec2 {
        SmallVec2::new_from(self.offset.x + x, self.offset.y + y)
    }
}

pub struct QuadTree {
    pub boundary: Rectangle,
    pub center_of_gravity: Vec2,
    pub tree_type: QuadTreeType,
}

pub enum QuadTreeType {
    Leaf {
        count: u8,
        children: Box<[Option<Particle>; 4]> },
    Root {
        total_mass: f32,
        children: Box<[QuadTree; 4]>,
    },
}

impl QuadTree {
    const MAX_CAPACITY: u8 = 4;
    pub fn new(boundary: Rectangle) -> Self {
        QuadTree {
            boundary,
            tree_type: QuadTreeType::Leaf{ count: 0, children: Box::from([None, None, None, None]) },
            center_of_gravity: Vec2::new(),
        }
    }

    pub fn get_total_mass(&self) -> f32 {
        match &self.tree_type {
            QuadTreeType::Leaf {count: _, children} => children
                .iter()
                .flatten()
                .fold(0.0, |a, particle| a + particle.weight),
            QuadTreeType::Root {
                total_mass,
                children: _,
            } => *total_mass,
        }
    }

    pub fn insert(&mut self, point: Particle) {
        match self.tree_type {
            QuadTreeType::Leaf{count, ref mut children} => {
                if count == QuadTree::MAX_CAPACITY {
                    self.subdivide();
                    self.insert(point);
                } else {
                    children[count as usize] = Some(point);
                }
            }
            QuadTreeType::Root {
                ref mut children,
                total_mass: _,
            } => {
                let half_height = self.boundary.height / 2.0;

                let hori_half = self.boundary.offset.x + half_height;
                let vert_half = self.boundary.offset.y + half_height;
                let north = point.position.y < vert_half as f64;
                let west = point.position.x < hori_half as f64;

                match north {
                    true => match west {
                        true => children[0].insert(point),
                        false => children[1].insert(point),
                    },
                    false => match west {
                        true => children[2].insert(point),
                        false => children[3].insert(point),
                    },
                };
            }
        }
    }

    fn subdivide(&mut self) {
        match &mut self.tree_type {
            QuadTreeType::Leaf{count, children} => {
                let new_height = self.boundary.height / 2.0;

                let mut new = QuadTree {
                    boundary: self.boundary.clone(),
                    tree_type: QuadTreeType::Root {
                        children: Box::from([
                            QuadTree::new(Rectangle::new(self.boundary.offset.clone(), new_height)),
                            QuadTree::new(Rectangle::new(
                                self.boundary.offset(new_height, 0.0),
                                new_height,
                            )),
                            QuadTree::new(Rectangle::new(
                                self.boundary.offset(0.0, new_height),
                                new_height,
                            )),
                            QuadTree::new(Rectangle::new(
                                self.boundary.offset(new_height, new_height),
                                new_height,
                            )),
                        ]),
                        total_mass: *count as f32,
                    },
                    center_of_gravity: self.center_of_gravity.clone(),
                };

                let current_children = mem::take(children);
                for p in current_children.into_iter().flatten() {
                    new.insert(p);
                }
                *self = new;
            }
            _ => {}
        }
    }

    pub fn calculate_gravity(&mut self) {
        match self.tree_type {
            QuadTreeType::Leaf{count, ref children} => {
                if count > 0 {
                    self.center_of_gravity = children
                        .iter()
                        .flatten()
                        .fold(Vec2::new(), |acc, part| acc.add(&part.position))
                        .div(count as f64);
                }
            }
            QuadTreeType::Root {
                ref mut children,
                ref mut total_mass,
            } => {
                children
                    .iter_mut()
                    .for_each(|child| child.calculate_gravity());

                let mass = children.iter().map(|child| child.get_total_mass()).sum();

                let big_thing = children.iter().fold(Vec2::new(), |acc, child| {
                    acc.add(&child.center_of_gravity.mul_32(child.get_total_mass()))
                });

                self.center_of_gravity = big_thing.div_32(mass);
                *total_mass = mass;
            }
        }
    }
}
