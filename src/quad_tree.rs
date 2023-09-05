use crate::Vec2;

pub enum QuadTree {
    Leaf {
        boundary: Rectangle,
        points: Vec<Vec2>,
    },
    Root {
        ne: Box<QuadTree>,
        se: Box<QuadTree>,
        sw: Box<QuadTree>,
        nw: Box<QuadTree>,
    },
}

impl QuadTree {
    const MAX_CAPACITY: usize = 4;

    pub fn new(boundary: Rectangle) -> Self {
        QuadTree::Leaf {
            boundary,
            points: Vec::new(),
        }
    }

    pub fn count(&self) -> usize {
        match self {
            QuadTree::Leaf {
                boundary: _,
                points,
            } => return points.len(),
            QuadTree::Root { ne, se, sw, nw } => {
                return ne.count() + se.count() + sw.count() + nw.count()
            }
        }
    }

    pub fn insert(&mut self, point: Vec2) -> Result<(), String> {
        match self {
            QuadTree::Leaf { boundary, points } => {
                if !boundary.contains(&point) {
                    return Err(String::from("Boundary doesn't contain point"));
                } else if points.len() == QuadTree::MAX_CAPACITY {
                    self.subdivide();
                    return self.insert(point);
                } else {
                    points.push(point);
                    return Ok(());
                }
            }
            QuadTree::Root { ne, se, sw, nw } => {
                if ne.insert(point).is_ok() {
                    return Ok(());
                } else if se.insert(point).is_ok() {
                    return Ok(());
                } else if sw.insert(point).is_ok() {
                    return Ok(());
                } else if nw.insert(point).is_ok() {
                    return Ok(());
                }
                return Err(());
            }
        }
    }

    fn subdivide(&mut self) {
        match self {
            QuadTree::Leaf { boundary, points } => {
                let new_width = boundary.width / 2.0;
                let new_height = boundary.height / 2.0;

                let mut new = QuadTree::Root {
                    ne: Box::new(QuadTree::new(Rectangle::new(
                        boundary.p0.offset(new_width, 0.0),
                        new_width,
                        new_height,
                    ))),
                    se: Box::new(QuadTree::new(Rectangle::new(
                        boundary.p0.offset(new_width, new_height),
                        new_width,
                        new_height,
                    ))),
                    sw: Box::new(QuadTree::new(Rectangle::new(
                        boundary.p0.offset(0.0, new_height),
                        new_width,
                        new_height,
                    ))),
                    nw: Box::new(QuadTree::new(Rectangle::new(
                        boundary.p0.offset(0.0, 0.0),
                        new_width,
                        new_height,
                    ))),
                };
                for p in points {
                    new.insert(*p).unwrap();
                }
                mem::replace(self, new);
            }
            _ => {}
        }
    }
}
