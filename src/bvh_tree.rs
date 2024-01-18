use pathfinder_geometry::vector::vec2f;

use crate::{SmallParticle, Vec2};
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
        children: Vec<BVHTree<'a>>,
        center_of_gravity: Vec2,
        boundary: Rectangle,
        total_mass: u32,
    },
    Leaf {
        children: &'a [SmallParticle],
        boundary: Rectangle,
    },
}

const TARGET_POINTS: usize = 64;

impl BVHTree<'_> {
    pub fn make_leaf(points: &[SmallParticle]) -> BVHTree {
        let mut min = vec2f(f32::MAX, f32::MAX);
        let mut max = vec2f(0.0, 0.0);
        for part in points.iter() {
            min = min.min(part.position);
            max = max.max(part.position);
        }

        let bounds = Rectangle {
            offset: min,
            size: max - min,
        };
        BVHTree::Leaf {
            children: points,
            boundary: bounds,
        }
    }

    pub fn from<'a>(points: &'a mut [SmallParticle]) -> BVHTree<'a> {
        let mut min = vec2f(f32::MAX, f32::MAX);
        let mut max = vec2f(0.0, 0.0);
        for part in points.iter() {
            min = min.min(part.position);
            max = max.max(part.position);
        }

        let bounds = Rectangle {
            offset: min,
            size: max - min,
        };
        let halved = bounds.size / 2.0;
        let (left, right): (&mut [SmallParticle], &mut [SmallParticle]);
        if halved.x() > halved.y() {
            let split_point = halved.x() + bounds.offset.x();
            (left, right) = partition(points, |part| part.position.x() > split_point);
        } else {
            let split_point = halved.y() + bounds.offset.y();
            (left, right) = partition(points, |part| part.position.y() > split_point);
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
            children: vec![left_leaf, right_leaf],
            center_of_gravity: vec2f(0.0, 0.0),
            boundary: bounds,
            total_mass: 0,
        }
    }

    pub fn get_center_of_gravity(&self) -> Vec2 {
        match *self {
            BVHTree::Leaf { children, boundary: _ } => {
                children
                    .iter()
                    .fold(vec2f(0.0, 0.0), |acc, part| acc + part.position)
                    / children.len() as f32
            },
            BVHTree::Root { children: _, center_of_gravity, boundary: _, total_mass: _ } => center_of_gravity
        }
    }

    pub fn get_total_mass(&self) -> u32 {
        match *self {
            BVHTree::Leaf { boundary: _, children } => children
                .iter()
                .fold(0, |a, particle| a + particle.weight),
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
            } => {},
            BVHTree::Root {
                ref mut children,
                ref mut total_mass,
                boundary: _,
                ref mut center_of_gravity
            } => {
                children
                    .iter_mut()
                    .for_each(|child| child.calculate_gravity());

                let mass = children
                    .iter()
                    .map(|child| child.get_total_mass())
                    .sum();

                let big_thing = children
                    .iter()
                    .fold(vec2f(0.0, 0.0), |acc, child| {
                        acc + (child.get_center_of_gravity() * child.get_total_mass() as f32)
                    });

                *center_of_gravity = big_thing / mass as f32;
                *total_mass = mass;
            }
        }
    }
}
