use crate::{Particle, Vec2};

#[derive(Clone)]
pub struct Rectangle {
    pub offset: Vec2,
    pub height: f64,
    pub height2: f64,
}

impl Rectangle {
    pub fn new(offset: Vec2, height: f64) -> Self {
        Self {
            offset,
            height,
            height2: height * height,
        }
    }

    pub fn contains(self: &Rectangle, other: &Vec2) -> bool {
        other.y > self.offset.y
            && other.x > self.offset.x
            && other.x < (self.offset.x + self.height)
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
    Leaf(Box<[Option<Particle>; 4]>),
    Root {
        total_mass: f64,
        children: Box<[QuadTree; 4]>,
    },
}

impl QuadTree {
    const MAX_CAPACITY: usize = 4;
    pub fn new(boundary: Rectangle) -> Self {
        QuadTree {
            boundary,
            tree_type: QuadTreeType::Leaf(Box::from([None, None, None, None])),
            center_of_gravity: Vec2::new(),
        }
    }

    pub fn get_total_mass(&self) -> f64 {
        match &self.tree_type {
            QuadTreeType::Leaf(points) => points.iter().flatten().count() as f64,
            QuadTreeType::Root {
                total_mass,
                children: _,
            } => *total_mass,
        }
    }

    pub fn insert(&mut self, point: Particle) {
        match self.tree_type {
            QuadTreeType::Leaf(ref mut points) => {
                let len = points.iter().flatten().count();
                if len == QuadTree::MAX_CAPACITY {
                    self.subdivide();
                    self.insert(point);
                } else {
                    points[len] = Some(point);
                }
            }
            QuadTreeType::Root {
                ref mut children,
                total_mass: _,
            } => {
                let half_height = self.boundary.height / 2.0;

                let hori_half = self.boundary.offset.x + half_height;
                let vert_half = self.boundary.offset.y + half_height;
                let north = point.position.y < vert_half;
                let west = point.position.x < hori_half;

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
        match &self.tree_type {
            QuadTreeType::Leaf(points) => {
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
                        total_mass: points.len() as f64,
                    },
                    center_of_gravity: self.center_of_gravity.clone(),
                };
                for p in points.iter().flatten() {
                    new.insert(p.clone());
                }
                *self = new;
            }
            _ => {}
        }
    }

    pub fn calculate_gravity(&mut self) {
        match self.tree_type {
            QuadTreeType::Leaf(ref points) => {
                let len = points.iter().flatten().count();
                if len > 0 {
                    self.center_of_gravity = points
                        .iter()
                        .flatten()
                        .fold(Vec2::new(), |acc, part| acc.add(&part.position))
                        .div(len as f64);
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
                    acc.add(&child.center_of_gravity.mul(child.get_total_mass()))
                });

                self.center_of_gravity = big_thing.div(mass);
                *total_mass = mass;
            }
        }
    }
}
