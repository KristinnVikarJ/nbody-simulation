use std::ops::{Add, Div};

use pathfinder_geometry::vector::vec2f;

use crate::{Vec2, Particle};
use partition::partition;

#[derive(Clone, Debug)]
pub struct Rectangle {
    pub offset: Vec2,
    pub size: Vec2,
}

impl Rectangle {
    pub fn contains(self: &Rectangle, other: &Vec2) -> bool {
        other.y() as f32 > self.offset.y()
            && (other.x() as f32) > self.offset.x()
            && (other.x() as f32) < (self.offset.x() + self.size.x())
            && (other.y() as f32) < (self.offset.y() + self.size.y())
    }
}

#[derive(Debug)]
pub enum BVHTree<'a> {
    Root {
        boundary: Rectangle,
        total_mass: u32,
        children: Box<[BVHTree<'a>; 2]>,
        center_of_gravity: Vec2,
    },
    Leaf {
        children: &'a [Particle],
        boundary: Rectangle,
    },
}

const TARGET_POINTS: usize = 64;

impl BVHTree<'_> {
    pub fn make_leaf(points: &[Particle]) -> BVHTree {
        let (min, max) = points.iter().fold(
            (vec2f(f32::MAX, f32::MAX), vec2f(0.0, 0.0)),
            |(min, max), p| (min.min(p.position), max.max(p.position)),
        );

        let bounds = Rectangle {
            offset: min,
            size: max - min,
        };
        BVHTree::Leaf {
            children: points,
            boundary: bounds,
        }
    }

    #[inline(always)]
    pub fn from<'a>(points: &'a mut [Particle]) -> BVHTree<'a> {
        let (min, max, sum) = points.iter().fold(
            (vec2f(f32::MAX, f32::MAX), vec2f(0.0, 0.0), vec2f(0.0, 0.0)),
            |(min, max, sum), p| (min.min(p.position), max.max(p.position), sum.add(p.position)),
        );

        let bounds = Rectangle {
            offset: min,
            size: max - min,
        };
        let halved = sum.div(points.len() as f32);

        let (left, right): (&mut [Particle], &mut [Particle]);
        let half_len = points.len() / 2;
        let hori = half_len.abs_diff(points.iter().filter(|p| p.position.x() > halved.x()).count());
        let vert = half_len.abs_diff(points.iter().filter(|p| p.position.y() > halved.y()).count());
        if vert > hori {
            (left, right) = partition(points, |part| part.position.x() > halved.x());
        } else {
            (left, right) = partition(points, |part| part.position.y() > halved.y());
        }
        let left_leaf = if left.len() > TARGET_POINTS {
            BVHTree::from(left)
        } else {
            BVHTree::make_leaf(left)
        };

        let right_leaf = if right.len() > TARGET_POINTS {
            BVHTree::from(right)
        } else {
            BVHTree::make_leaf(right)
        };

        BVHTree::Root {
            children: Box::from([left_leaf, right_leaf]),
            center_of_gravity: vec2f(0.0, 0.0),
            boundary: bounds,
            total_mass: 0,
        }
    }

    pub fn get_center_of_gravity(&self) -> Vec2 {
        match *self {
            BVHTree::Leaf {
                children,
                boundary: _,
            } => {
                children
                    .iter()
                    .fold(vec2f(0.0, 0.0), |acc, part| acc + part.position)
                    / children.len() as f32
            }
            BVHTree::Root {
                children: _,
                center_of_gravity,
                boundary: _,
                total_mass: _,
            } => center_of_gravity,
        }
    }

    pub fn get_total_mass(&self) -> u32 {
        match *self {
            BVHTree::Leaf {
                boundary: _,
                children,
            } => children.iter().fold(0, |a, particle| a + particle.weight),
            BVHTree::Root {
                center_of_gravity: _,
                boundary: _,
                total_mass,
                children: _,
            } => total_mass,
        }
    }

    pub fn calculate_gravity(&mut self) {
        match self {
            BVHTree::Leaf {
                boundary: _,
                children: _,
            } => {}
            BVHTree::Root {
                ref mut children,
                ref mut total_mass,
                boundary: _,
                ref mut center_of_gravity,
            } => {
                children[0].calculate_gravity();
                children[1].calculate_gravity();

                let mass = children[0].get_total_mass() + children[1].get_total_mass();

                let big_thing = (children[0].get_center_of_gravity()
                    * children[0].get_total_mass() as f32)
                    + (children[1].get_center_of_gravity() * children[1].get_total_mass() as f32);

                *center_of_gravity = big_thing / mass as f32;
                *total_mass = mass;
            }
        }
    }
}
