package arow

type Arow struct {
	dimension int
	mean      []float64
	cov       []float64

	r float64
}

func NewArow(features int, param float64) *Arow {
	a := &Arow{
		mean:      make([]float64, features),
		cov:       make([]float64, features),
		dimension: features,
		r:         param,
	}

	for i := 0; i < features; i++ {
		a.cov[i] = 1.0
	}
	return a
}

func (self *Arow) confidence(f []float64) float64 {
	confidence := float64(0)
	for i := 0; i < len(f); i++ {
		confidence += self.cov[i] * f[i] * f[i]
	}

	return confidence
}

func (self *Arow) margin(f []float64) float64 {
	margin := float64(0)
	for i := 0; i < len(f); i++ {
		margin += self.mean[i] * f[i]
	}
	return margin
}

func (self *Arow) Predict(f []float64) int {
	margin := self.margin(f)
	if margin > 0 {
		return 1
	} else {
		return -1
	}
}

func (self *Arow) Update(f []float64, label int) int {
	margin := self.margin(f)
	if label > 0 {
		label = 1
	} else {
		label = -1
	}

	if margin*float64(label) >= 1 {
		return 0
	}

	confidence := self.confidence(f)
	beta := 1.0 / (confidence + self.r)
	alpha := (1.0 - float64(label)*margin) * beta

	for i := 0; i < len(f); i++ {
		self.mean[i] += alpha * float64(label) * self.cov[i] * f[i]
	}
	for i := 0; i < len(f); i++ {
		self.cov[i] = 1.0 / ((1.0 / self.cov[i]) + f[i]*f[i]/self.r)
	}

	if margin*float64(label) < 0 {
		return 1
	} else {
		return 0
	}
}
