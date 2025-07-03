
// track min, max, mean, median, 25p, 75p of a stream of numbers using approximations
pub struct StatTracker<T: Ord> {
    pub stream: quantiles::greenwald_khanna::Stream<T>,
    pub min: T,
    pub max: T,
    pub sum: T,
    pub count: u32,
}

impl <T: Ord + Clone + num::ToPrimitive + num::Zero + std::ops::AddAssign + num::Bounded + Default> StatTracker<T> {
    pub fn new() -> StatTracker<T> {
        StatTracker {
            stream: quantiles::greenwald_khanna::Stream::new(0.01),
            min: T::max_value(),
            max: T::min_value(),
            sum: T::default(),
            count: 0,
        }
    }

    pub fn reset(&mut self) {
        self.stream = quantiles::greenwald_khanna::Stream::new(0.01);
        self.min = T::max_value();
        self.max = T::min_value();
        self.sum = T::default();
        self.count = 0;
    }

    pub fn insert(&mut self, val: T) {
        self.stream.insert(val.clone());
        self.min = std::cmp::min(self.min.clone(), val.clone());
        self.max = std::cmp::max(self.max.clone(), val.clone());
        self.sum = self.sum.clone() + val.clone();
        self.count += 1;
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum.to_f64().unwrap() / self.count as f64
    }

    pub fn median(&self) -> T {
        self.stream.quantile(0.5).clone()
    }

    pub fn percentile(&self, p: f64) -> T {
        self.stream.quantile(p).clone()
    }

    pub fn p25(&self) -> T {
        self.percentile(0.25)
    }

    pub fn p75(&self) -> T {
        self.percentile(0.75)
    }

    pub fn min(&self) -> T {
        if self.min == T::max_value() {
            return T::default();
        }
        self.min.clone()
    }

    pub fn max(&self) -> T {
        if self.max == T::min_value() {
            return T::default();
        }
        self.max.clone()
    }

    pub fn count(&self) -> u32 {
        self.count
    }
}
