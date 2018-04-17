package graphprop

import (
	"math"
	"sync"
)

type result struct {
	vi    int
	alpha map[int]float64
}

func (c *Corpus) propSingleSeed(vi int, T int) result {
	alpha := make(map[int]float64)
	for vj, w := range c.W[vi] {
		alpha[vj] = w
	}
	F := map[int]struct{}{vi: {}}
	for t := 0; t < T; t++ {
		updateF := make(map[int]struct{})
		for vk := range F {
			for vj, wkj := range c.W[vk] {
				aikj := alpha[vk] * wkj
				if alpha[vj] < aikj {
					alpha[vj] = aikj
				}
				updateF[vj] = struct{}{}
			}
		}
		for v := range updateF {
			F[v] = struct{}{}
		}
	}
	return result{
		vi:    vi,
		alpha: alpha,
	}
}

func (c *Corpus) GenLexicon(T int, gamma float64, nWorkers int) map[string]float64 {
	seeds := make(chan int, nWorkers)
	results := make(chan result, nWorkers)

	// Spin up nWorkers goroutines to propagate sentiment from single nodes
	var wg sync.WaitGroup
	for i := 0; i < nWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for vi := range seeds {
				res := c.propSingleSeed(vi, T)
				results <- res
			}
		}()
	}
	// Close results channel when all the workers are done.
	go func() {
		wg.Wait()
		close(results)
	}()
	// Push all the seed nodes into the queue, we'll sort out polarity later
	go func() {
		for pi := range c.P {
			seeds <- pi
		}
		for ni := range c.N {
			seeds <- ni
		}
		close(seeds)
	}()

	// Aggregate results across N and P
	polPos := make(map[int]float64)
	polNeg := make(map[int]float64)
	for res := range results {
		polarity := polPos
		if _, ok := c.N[res.vi]; ok {
			polarity = polNeg
		}
		for vj, aij := range res.alpha {
			polarity[vj] += aij
		}
	}

	// Calculate normalization factor beta
	totalP := 0.0
	totalN := 0.0

	for _, a := range polPos {
		totalP += a
	}
	for _, a := range polNeg {
		totalN += a
	}
	beta := totalP / totalN

	// Calculate the final polarity and threshold using gamma
	pol := make(map[string]float64)
	for item, id := range c.IDMap {
		ipol := polPos[id] - beta*polNeg[id]
		if math.Abs(ipol) < gamma {
			continue
		}
		pol[item] = ipol
	}
	return pol
}
