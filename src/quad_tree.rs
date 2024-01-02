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
        children: Box<[Option<Particle>; 4]>,
    },
    Root {
        total_mass: u32,
        children: Box<[Option<QuadTree>; 4]>,
    },
}

impl QuadTree {
    const MAX_CAPACITY: u8 = 4;
    pub fn new(boundary: Rectangle) -> Self {
        QuadTree {
            boundary,
            tree_type: QuadTreeType::Leaf {
                count: 0,
                children: Box::from([None, None, None, None]),
            },
            center_of_gravity: Vec2::new(),
        }
    }

    pub fn empty(&mut self) -> u32 {
        match &mut self.tree_type {
            QuadTreeType::Leaf { count, children } => {
                *count = 0;
                children.iter_mut().for_each(|child| *child = None);
                return 1;
            },
            QuadTreeType::Root { total_mass, children } => {
                if *total_mass == 0 {
                    return 1;
                }
                *total_mass = 0;
                let mut sum = 0;
                for child in children.iter_mut().flatten() {
                    sum += child.empty();
                }
                return sum + 1;
            }
        }
    }

    // Must be ran after calculate_gravity
    pub fn prune(&mut self) -> u32 {
        let mut sum = 0;
        match &mut self.tree_type {
            QuadTreeType::Leaf { count: _, children: _ } => {},
            QuadTreeType::Root { total_mass: _, children } => {
                for child in children.iter_mut() {
                    let mut child_empty = false;
                    if let Some(inner_child) = child {
                        match &mut inner_child.tree_type {
                            QuadTreeType::Leaf { count , children: _ } => {
                                if *count == 0 {
                                    child_empty = true
                                }
                            },
                            QuadTreeType::Root { total_mass, children: _ } => {
                                if *total_mass == 0 {
                                    child_empty = true;
                                } else {
                                    sum += inner_child.prune();
                                }
                            }
                        }
                    }
                    if child_empty {
                        *child = None;
                        sum += 1;
                    }
                }
            }
        }
        return sum;
    }

    pub fn get_total_mass(&self) -> u32 {
        match &self.tree_type {
            QuadTreeType::Leaf { count: _, children } => children
                .iter()
                .flatten()
                .fold(0, |a, particle| a + particle.weight),
            QuadTreeType::Root {
                total_mass,
                children: _,
            } => *total_mass,
        }
    }

    pub fn insert(&mut self, point: Particle) {
        match self.tree_type {
            QuadTreeType::Leaf {
                ref mut count,
                ref mut children,
            } => {
                if *count == QuadTree::MAX_CAPACITY {
                    self.subdivide();
                    self.insert(point);
                } else {
                    children[*count as usize] = Some(point);
                    *count += 1;
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
                        true => {
                            if children[0].is_none() {
                                children[0] = Some(QuadTree::new(Rectangle::new(
                                    self.boundary.offset.clone(),
                                    half_height,
                                )));
                            }
                            children[0].as_mut().unwrap().insert(point)
                        }
                        false => {
                            if children[1].is_none() {
                                children[1] = Some(QuadTree::new(Rectangle::new(
                                    self.boundary.offset(half_height, 0.0),
                                    half_height,
                                )));
                            }
                            children[1].as_mut().unwrap().insert(point)
                        }
                    },
                    false => match west {
                        true => {
                            if children[2].is_none() {
                                children[2] = Some(QuadTree::new(Rectangle::new(
                                    self.boundary.offset(0.0, half_height),
                                    half_height,
                                )));
                            }
                            children[2].as_mut().unwrap().insert(point)
                        }
                        false => {
                            if children[3].is_none() {
                                children[3] = Some(QuadTree::new(Rectangle::new(
                                    self.boundary.offset(half_height, half_height),
                                    half_height,
                                )));
                            }
                            children[3].as_mut().unwrap().insert(point)
                        }
                    },
                };
            }
        }
    }

    fn subdivide(&mut self) {
        if let QuadTreeType::Leaf { count, children } = &mut self.tree_type {
            let mut new = QuadTree {
                boundary: self.boundary.clone(),
                tree_type: QuadTreeType::Root {
                    children: Box::from([None, None, None, None]),
                    total_mass: *count as u32,
                },
                center_of_gravity: Vec2::new(),
            };

            let current_children = mem::take(children);
            for p in current_children.into_iter() {
                new.insert(p.unwrap());
            }
            *self = new;
        }
    }

    pub fn calculate_gravity(&mut self) {
        match self.tree_type {
            QuadTreeType::Leaf {
                count,
                ref children,
            } => {
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
                    .flatten()
                    .for_each(|child| child.calculate_gravity());

                let mass = children
                    .iter()
                    .flatten()
                    .map(|child| child.get_total_mass())
                    .sum();

                let big_thing = children.iter().flatten().fold(Vec2::new(), |acc, child| {
                    acc.add(&child.center_of_gravity.mul_32(child.get_total_mass() as f32))
                });

                self.center_of_gravity = big_thing.div_32(mass as f32);
                *total_mass = mass;
            }
        }
    }
}
