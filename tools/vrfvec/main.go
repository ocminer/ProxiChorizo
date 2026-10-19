// vrfvec prints ECVRF-EDWARDS25519-SHA512-TAI test vectors (seed, key, alpha,
// intermediate points and beta) as JSON lines, for validating the GPU miner
// against the node's own util/vrf implementation.
package main

import (
	"crypto/ed25519"
	"crypto/sha512"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"

	"filippo.io/edwards25519"
	"github.com/lunfardo314/proxima/util/vrf"
)

type vec struct {
	Seed  string `json:"seed"`
	Pk    string `json:"pk"`
	X     string `json:"x"` // clamped secret scalar, 32 bytes little-endian
	Alpha string `json:"alpha"`
	Ctr   int    `json:"ctr"`
	H     string `json:"h"`
	Gamma string `json:"gamma"`
	CG    string `json:"cg"` // 8*Gamma
	Beta  string `json:"beta"`
	TZ    int    `json:"tz"` // trailing zero bits of beta
}

func encode(pk, alpha []byte) (*edwards25519.Point, int) {
	id := edwards25519.NewIdentityPoint()
	for ctr := 0; ctr < 256; ctr++ {
		hb := append([]byte{0x03, 0x01}, pk...)
		hb = append(hb, alpha...)
		hb = append(hb, byte(ctr), 0x00)
		d := sha512.Sum512(hb)
		p, err := new(edwards25519.Point).SetBytes(d[:32])
		if err != nil {
			continue
		}
		p.MultByCofactor(p)
		if p.Equal(id) == 1 {
			continue
		}
		return p, ctr
	}
	panic("encode failed")
}

func tz(b []byte) int {
	n := 0
	for i := len(b) - 1; i >= 0; i-- {
		if b[i] == 0 {
			n += 8
			continue
		}
		for x := b[i]; x&1 == 0; x >>= 1 {
			n++
		}
		return n
	}
	return n
}

func main() {
	count := 16
	seedSrc := "proxima gpu vectors"
	if len(os.Args) > 1 {
		seedSrc = os.Args[1]
	}
	hs := sha512.Sum512([]byte(seedSrc))
	seed := make([]byte, 32)
	copy(seed, hs[:32])
	sk := ed25519.NewKeyFromSeed(seed)
	pk := sk.Public().(ed25519.PublicKey)
	h := sha512.Sum512(seed)
	xb := make([]byte, 32)
	copy(xb, h[:32])
	xb[0] &= 248
	xb[31] &= 127
	xb[31] |= 64
	x, err := edwards25519.NewScalar().SetBytesWithClamping(h[:32])
	if err != nil {
		panic(err)
	}
	prover, err := vrf.NewProver(sk)
	if err != nil {
		panic(err)
	}
	for i := 0; i < count; i++ {
		a := sha512.Sum512([]byte(fmt.Sprintf("%s alpha %d", seedSrc, i)))
		alpha := a[:45] // 37-byte prefix + 8-byte nonce, like a real mine message
		beta, _, err := prover.Output(alpha)
		if err != nil {
			panic(err)
		}
		H, ctr := encode(pk, alpha)
		gamma := new(edwards25519.Point).ScalarMult(x, H)
		cg := new(edwards25519.Point).MultByCofactor(gamma)
		// cross-check with the full prove/verify path
		pi, err := vrf.Prove(sk, alpha)
		if err != nil {
			panic(err)
		}
		b2, err := vrf.Verify(pk, alpha, pi)
		if err != nil || hex.EncodeToString(b2) != hex.EncodeToString(beta) {
			panic("verify mismatch")
		}
		v := vec{
			Seed: hex.EncodeToString(seed), Pk: hex.EncodeToString(pk), X: hex.EncodeToString(xb),
			Alpha: hex.EncodeToString(alpha), Ctr: ctr, H: hex.EncodeToString(H.Bytes()),
			Gamma: hex.EncodeToString(gamma.Bytes()), CG: hex.EncodeToString(cg.Bytes()),
			Beta: hex.EncodeToString(beta), TZ: tz(beta),
		}
		j, _ := json.Marshal(v)
		fmt.Println(string(j))
	}
}
