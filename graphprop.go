package graphprop

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path"
	"strconv"
	"strings"
)

type Corpus struct {
	IDMap map[string]int
	W     map[int]map[int]float64
	P     map[int]struct{}
	N     map[int]struct{}
}

func loadSeedFile(fp string, idmap map[string]int) (map[int]struct{}, error) {
	raw, err := ioutil.ReadFile(fp)
	if err != nil {
		return nil, err
	}
	seeds := make(map[int]struct{})
	for _, item := range strings.Split(string(raw), "\n") {
		seeds[idmap[item]] = struct{}{}
	}
	return seeds, nil
}

func loadSims(fp string) (map[string]int, map[int]map[int]float64, error) {

	f, err := os.Open(fp)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	idMap := make(map[string]int)
	W := make(map[int]map[int]float64)

	for scanner.Scan() {
		fields := bytes.Split(scanner.Bytes(), []byte(","))
		tid1, ok := idMap[string(fields[0])]
		if !ok {
			tid1 = len(idMap)
			idMap[string(fields[0])] = tid1
		}
		tid2, ok := idMap[string(fields[1])]
		if !ok {
			tid2 = len(idMap)
			idMap[string(fields[1])] = tid2
		}
		sim, err := strconv.ParseFloat(string(fields[2]), 64)
		if err != nil {
			return nil, nil, err

		}
		if _, ok := W[tid1]; !ok {
			W[tid1] = make(map[int]float64)
		}
		if _, ok := W[tid2]; !ok {
			W[tid2] = make(map[int]float64)
		}
		W[tid1][tid2] = sim
		W[tid2][tid1] = sim
	}
	if err := scanner.Err(); err != nil {
		return nil, nil, err
	}
	return idMap, W, nil
}

func LoadCorpus(dir string) (*Corpus, error) {
	WPath := path.Join(dir, "similarity.csv")
	idmap, W, err := loadSims(WPath)
	if err != nil {
		return nil, err
	}
	PPath := path.Join(dir, "positive.txt")

	P, err := loadSeedFile(PPath, idmap)
	if err != nil {
		return nil, err
	}

	NPath := path.Join(dir, "negative.txt")
	N, err := loadSeedFile(NPath, idmap)
	if err != nil {
		return nil, err
	}
	return &Corpus{
		IDMap: idmap,
		P:     P,
		N:     N,
		W:     W,
	}, nil
}

func (c *Corpus) Propagate(T int, gamma float64) {
	// Propagate positive sentiment

	// This should be a copy of c.W initially.
	alpha := make(map[[2]int]float64)
	for vi, neighbors := range c.W {
		for vj, wij := range neighbors {
			alpha[[2]int{vi, vj}] = wij
		}
	}

	for vi := range c.P {
		F := map[int]struct{}{vi: {}}
		for t := 0; t < T; t++ {
			for vk := range F {
				for vj, wkj := range c.W[vk] {
					aij := alpha[[2]int{vi, vj}]
					aikj := alpha[[2]int{vi, vk}] * wkj
					if aikj > aij {
						alpha[[2]int{vi, vj}] = aikj
					}
					F[vj] = struct{}{}
				}
			}
		}
	}
	polP := make([]float64, len(c.IDMap))

	for _, vj := range c.IDMap {
		for vi := range c.P {
			polP[vj] += alpha[[2]int{vi, vj}]
		}
	}
	fmt.Printf("%v\n", polP)

	// Repeat the process with N

	// This should be a copy of c.W initially.
	alpha = make(map[[2]int]float64)
	for vi, neighbors := range c.W {
		for vj, wij := range neighbors {
			alpha[[2]int{vi, vj}] = wij
		}
	}

	for vi := range c.N {
		F := map[int]struct{}{vi: {}}
		for t := 0; t < T; t++ {
			for vk := range F {
				for vj, wkj := range c.W[vk] {
					aij := alpha[[2]int{vi, vj}]
					aikj := alpha[[2]int{vi, vk}] * wkj
					if aikj > aij {
						alpha[[2]int{vi, vj}] = aikj
					}
					F[vj] = struct{}{}
				}
			}
		}
	}
	polN := make([]float64, len(c.IDMap))

	for _, vj := range c.IDMap {
		for vi := range c.N {
			polN[vj] += alpha[[2]int{vi, vj}]
		}
	}

	fmt.Printf("%v\n", polN)

	// Calculate the normalization factor beta
	totalP := 0.0
	for _, p := range polP {
		totalP += p
	}
	totalN := 0.0
	for _, p := range polN {
		totalN += p
	}
	beta := totalP / totalN

	// Generate the polarity scores and drop any below gamma
	pol := make([]float64, len(c.IDMap))
	for i := 0; i < len(c.IDMap); i++ {
		pol[i] = polP[i] - beta*polN[i]
		if math.Abs(pol[i]) < gamma {
			pol[i] = 0
		}
	}
	fmt.Printf("pol: %v\n", pol)
}
