package core

import "testing"

// Mirrors the Rust unit test: pgvector literals must match f32::to_string
// formatting (fixed notation, shortest round-trip).
func TestPgvectorLiteral(t *testing.T) {
	cases := []struct {
		in   []float32
		want string
	}{
		{[]float32{}, "[]"},
		{[]float32{1.0, 0.0, -1.0}, "[1,0,-1]"},
		{[]float32{0.5, 0.25}, "[0.5,0.25]"},
		{[]float32{0.000031}, "[0.000031]"}, // fixed notation, no exponent
	}
	for _, c := range cases {
		if got := ToPgvectorLiteral(c.in); got != c.want {
			t.Errorf("ToPgvectorLiteral(%v) = %q, want %q", c.in, got, c.want)
		}
	}
}
