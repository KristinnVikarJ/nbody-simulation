use std::{mem, hint};

use pathfinder_geometry::vector::vec2f;

use crate::{SmallParticle, Vec2};

#[derive(Clone)]
pub struct Rectangle {
    pub offset: Vec2,
    pub height: f32,
    pub height2: f32,
}

impl Rectangle {
    pub fn new(offset: Vec2, height: f32) -> Self {
        Self {
            offset,
            height,
            height2: height * height,
        }
    }

    pub fn contains(self: &Rectangle, other: &Vec2) -> bool {
        other.y() as f32 > self.offset.y()
            && (other.x() as f32) > self.offset.x()
            && (other.x() as f32) < (self.offset.x() + self.height)
            && (other.y() as f32) < (self.offset.y() + self.height)
    }

    pub fn offset(self: &Rectangle, x: f32, y: f32) -> Vec2 {
        vec2f(self.offset.x() + x, self.offset.y() + y)
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
        children: Box<[Option<SmallParticle>; 8]>,
    },
    Root {
        flags: u8,
        total_mass: u32,
        children: Box<[Option<QuadTree>; 4]>,
    },
}

impl QuadTree {
    const MAX_CAPACITY: u8 = 8;
    pub fn new(boundary: Rectangle) -> Self {
        QuadTree {
            boundary,
            tree_type: QuadTreeType::Leaf {
                count: 0,
                children: Box::from([None, None, None, None, None, None, None, None]),
            },
            center_of_gravity: vec2f(0.0, 0.0),
        }
    }

    pub fn empty(&mut self) -> u32 {
        match &mut self.tree_type {
            QuadTreeType::Leaf { count, children } => {
                *count = 0;
                children.iter_mut().for_each(|child| *child = None);
                1
            }
            QuadTreeType::Root {
                flags: _,
                total_mass,
                children,
            } => {
                if *total_mass == 0 {
                    return 1;
                }
                *total_mass = 0;
                let mut sum = 0;
                for child in children.iter_mut().flatten() {
                    sum += child.empty();
                }
                sum + 1
            }
        }
    }

    // Must be ran after calculate_gravity
    // TODO: if particle count in root (with 4 leaves) is < capacity
    // Merge them together, else we get thousands of near empty leaves
    pub fn prune(&mut self) -> u32 {
        let mut sum = 0;
        match &mut self.tree_type {
            QuadTreeType::Leaf {
                count: _,
                children: _,
            } => {}
            QuadTreeType::Root {
                flags,
                total_mass: _,
                children,
            } => {
                for (i, child) in children.iter_mut().enumerate() {
                    let mut child_empty = false;
                    if let Some(inner_child) = child {
                        match &mut inner_child.tree_type {
                            QuadTreeType::Leaf { count, children: _ } => {
                                if *count == 0 {
                                    child_empty = true
                                }
                            }
                            QuadTreeType::Root {
                                flags: _,
                                total_mass,
                                children: _,
                            } => {
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
                        *flags ^= 1 << i;
                        sum += 1;
                    }
                }
            }
        }
        sum
    }

    pub fn get_total_mass(&self) -> u32 {
        match &self.tree_type {
            QuadTreeType::Leaf { count: _, children } => children
                .iter()
                .flatten()
                .fold(0, |a, particle| a + particle.weight),
            QuadTreeType::Root {
                flags: _,
                total_mass,
                children: _,
            } => *total_mass,
        }
    }

    pub fn insert(&mut self, point: SmallParticle) {
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
                ref mut flags,
                ref mut children,
                total_mass: _,
            } => {
                let half_height = self.boundary.height / 2.0;

                let hori_half = self.boundary.offset.x() + half_height;
                let vert_half = self.boundary.offset.y() + half_height;
                let north = point.position.y() > vert_half;
                let west = point.position.x() > hori_half;
                
                let child = ((north as u8) << 1) + west as u8;

                if *flags & 1 << child == 0 {
                    let offset = match child {
                        0 => self.boundary.offset.clone(),
                        1 => self.boundary.offset(half_height, 0.0),
                        2 => self.boundary.offset(0.0, half_height),
                        3 => self.boundary.offset(half_height, half_height),
                        _ => unreachable!()
                    };

                    children[child as usize] = Some(QuadTree::new(Rectangle::new(
                        offset,
                        half_height,
                    )));

                    *flags |= 1 << child;
                }
                
                // We're guaranteed to know that children[child] exists, since the flag is set.
                unsafe {
                    match &mut children[child as usize] {
                        Some(child) => {
                            child.insert(point);
                        },
                        None => {
                            hint::unreachable_unchecked()
                        }
                    }
                }
            }
        }
    }

    fn subdivide(&mut self) {
        if let QuadTreeType::Leaf { count, children } = &mut self.tree_type {
            let mut new = QuadTree {
                boundary: self.boundary.clone(),
                tree_type: QuadTreeType::Root {
                    flags: 0,
                    children: Box::from([None, None, None, None]),
                    total_mass: *count as u32,
                },
                center_of_gravity: vec2f(0.0, 0.0),
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
                        .fold(vec2f(0.0, 0.0), |acc, part| acc + part.position)
                        / count as f32;
                }
            }
            QuadTreeType::Root {
                flags: _,
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

                let big_thing = children
                    .iter()
                    .flatten()
                    .fold(vec2f(0.0, 0.0), |acc, child| {
                        acc + (child.center_of_gravity * child.get_total_mass() as f32)
                    });

                self.center_of_gravity = big_thing / mass as f32;
                *total_mass = mass;
            }
        }
    }
}
