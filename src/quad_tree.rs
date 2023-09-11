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
            height2: height*height,
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
    Leaf {
        points: Vec<Particle>,
        sum_vec: Vec2,
    },
    Root {
        total_mass: f64,
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
                points: Vec::with_capacity(Self::MAX_CAPACITY),
                sum_vec: Vec2::new(),
            },
            center_of_gravity: Vec2::new(),
        }
    }

    pub fn get_total_mass(&self) -> f64 {
        match &self.tree_type {
            QuadTreeType::Leaf { points, sum_vec: _ } => points.len() as f64,
            QuadTreeType::Root {
                total_mass,
                ne: _,
                se: _,
                sw: _,
                nw: _,
            } => *total_mass,
        }
    }

    pub fn insert(&mut self, point: Particle) {
        match self.tree_type {
            QuadTreeType::Leaf {
                ref mut points,
                ref mut sum_vec,
            } => {
                if points.len() == QuadTree::MAX_CAPACITY {
                    self.subdivide();
                    self.insert(point);
                } else {
                    *sum_vec = sum_vec.add(&point.position);
                    points.push(point);
                }
            }
            QuadTreeType::Root {
                ref mut ne,
                ref mut se,
                ref mut sw,
                ref mut nw,
                total_mass: _,
            } => {
                let half_height = self.boundary.height / 2.0;

                let hori_half = self.boundary.offset.x + half_height;
                let vert_half = self.boundary.offset.y + half_height;
                let north = point.position.y < vert_half;
                let west = point.position.x < hori_half;

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
            }
        }
    }

    fn subdivide(&mut self) {
        match &self.tree_type {
            QuadTreeType::Leaf { points, sum_vec: _ } => {
                let new_height = self.boundary.height / 2.0;

                let mut new = QuadTree {
                    boundary: self.boundary.clone(),
                    tree_type: QuadTreeType::Root {
                        ne: Box::new(QuadTree::new(Rectangle::new(
                            self.boundary.offset(new_height, 0.0),
                            new_height,
                        ))),
                        se: Box::new(QuadTree::new(Rectangle::new(
                            self.boundary.offset(new_height, new_height),
                            new_height,
                        ))),
                        sw: Box::new(QuadTree::new(Rectangle::new(
                            self.boundary.offset(0.0, new_height),
                            new_height,
                        ))),
                        nw: Box::new(QuadTree::new(Rectangle::new(
                            self.boundary.offset.clone(),
                            new_height,
                        ))),
                        total_mass: points.len() as f64,
                    },
                    center_of_gravity: self.center_of_gravity.clone(),
                };
                for p in points {
                    new.insert(p.clone());
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
                    + sw.center_of_gravity.mul(sw.get_total_mass())
                    + nw.center_of_gravity.mul(nw.get_total_mass());

                self.center_of_gravity = big_thing.div(mass);
                *total_mass = mass;
            }
        }
    }
}
