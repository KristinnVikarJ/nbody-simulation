use pathfinder_geometry::vector::vec2f;

use crate::{Particle, SmallParticle, Vec2};

#[derive(Clone, Debug)]
pub struct Rectangle {
    pub offset: Vec2,
    pub size: Vec2,
}

#[derive(Debug)]
pub enum BVHTree {
    Root {
        children: Vec<BVHTree>,
        center_of_gravity: Vec2,
        boundary: Rectangle,
        total_mass: u32,
    },
    Leaf {
        children: Vec<SmallParticle>,
        boundary: Rectangle,
    },
}

const TARGET_POINTS: usize = 32;

impl BVHTree {
    pub fn make_leaf(points: Vec<SmallParticle>) -> BVHTree {
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

    pub fn from(points: Vec<SmallParticle>) -> BVHTree {
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
        let (left, right): (Vec<SmallParticle>, Vec<SmallParticle>);
        if halved.x() > halved.y() {
            let split_point = halved.x() + bounds.offset.x();
            (left, right) = points
                .iter()
                .cloned()
                .partition(|part| part.position.x() > split_point);
        } else {
            let split_point = halved.y() + bounds.offset.y();
            (left, right) = points
                .iter()
                .cloned()
                .partition(|part| part.position.y() > split_point);
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
}
