package graphprop

import (
	"bufio"
	"bytes"
	"io/ioutil"
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
