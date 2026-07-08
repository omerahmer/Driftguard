package core

// Faithful port of the diff pipeline behind Rust's
// similar::capture_diff_slices(Algorithm::Myers, …) — which is
// Compact(Replace(Capture)) over Myers' divide-and-conquer diff
// (similar 2.7.0: src/algorithms/{myers,compact,replace}.rs).
//
// Ported line-by-line ON PURPOSE, including tie-breaking behavior: the diff
// ops and changed/removed spans are persisted (prompt_versions.diff_from_parent)
// and embedded for eval selection, so the Go rewrite must reproduce the Rust
// output exactly. Parity is pinned by golden vectors (segment_test.go) and by
// recomputing every stored diff in the DB. Don't "simplify" the algorithm —
// an equally-valid LCS with different anchors breaks span-level parity.

type diffTag int

const (
	tagEqual diffTag = iota
	tagDelete
	tagInsert
)

// rangeOp mirrors similar::DiffOp (Equal / Delete / Insert). For Equal the
// length is duplicated into both oldLen and newLen; Delete has newLen == 0,
// Insert has oldLen == 0 — matching each variant's range semantics.
type rangeOp struct {
	tag      diffTag
	oldIndex int
	oldLen   int
	newIndex int
	newLen   int
}

func (op rangeOp) oldRange() (int, int) { return op.oldIndex, op.oldIndex + op.oldLen }
func (op rangeOp) newRange() (int, int) { return op.newIndex, op.newIndex + op.newLen }
func (op rangeOp) isEmpty() bool        { return op.oldLen <= 0 && op.newLen <= 0 }

func (op *rangeOp) adjust(offAdj int, offSub bool, lenAdj int, lenSub bool) {
	mod := func(v *int, adj int, sub bool) {
		if sub {
			*v -= adj
		} else {
			*v += adj
		}
	}
	mod(&op.oldIndex, offAdj, offSub)
	mod(&op.newIndex, offAdj, offSub)
	switch op.tag {
	case tagEqual:
		mod(&op.oldLen, lenAdj, lenSub)
		mod(&op.newLen, lenAdj, lenSub)
	case tagDelete:
		mod(&op.oldLen, lenAdj, lenSub)
	case tagInsert:
		mod(&op.newLen, lenAdj, lenSub)
	}
}

func (op *rangeOp) shiftLeft(n int)   { op.adjust(n, true, 0, false) }
func (op *rangeOp) shiftRight(n int)  { op.adjust(n, false, 0, false) }
func (op *rangeOp) growLeft(n int)    { op.adjust(n, true, n, false) }
func (op *rangeOp) growRight(n int)   { op.adjust(0, false, n, false) }
func (op *rangeOp) shrinkLeft(n int)  { op.adjust(0, false, n, true) }
func (op *rangeOp) shrinkRight(n int) { op.adjust(n, false, n, true) }

// myersDiff runs the full capture_diff_slices pipeline and returns range ops
// (post-Compact cleanup), ready for Replace-style flattening.
func myersDiff(old, new []string) []rangeOp {
	m := &myersState{old: old, new: new}
	maxD := maxD(len(old), len(new))
	m.vf = newVArr(maxD)
	m.vb = newVArr(maxD)
	m.conquer(0, len(old), 0, len(new))
	cleanupDiffOps(old, new, &m.ops)
	return m.ops
}

type myersState struct {
	old, new []string
	vf, vb   *vArr
	ops      []rangeOp
}

func (m *myersState) equal(oldIndex, newIndex, len int) {
	m.ops = append(m.ops, rangeOp{tag: tagEqual, oldIndex: oldIndex, oldLen: len, newIndex: newIndex, newLen: len})
}

func (m *myersState) delete(oldIndex, oldLen, newIndex int) {
	m.ops = append(m.ops, rangeOp{tag: tagDelete, oldIndex: oldIndex, oldLen: oldLen, newIndex: newIndex})
}

func (m *myersState) insert(oldIndex, newIndex, newLen int) {
	m.ops = append(m.ops, rangeOp{tag: tagInsert, oldIndex: oldIndex, newIndex: newIndex, newLen: newLen})
}

// vArr is similar's V: endpoints of the furthest-reaching D-paths, indexed by
// diagonal k which can be negative, hence the offset wrapper.
type vArr struct {
	offset int
	v      []int
}

func newVArr(maxD int) *vArr { return &vArr{offset: maxD, v: make([]int, 2*maxD)} }

func (a *vArr) get(k int) int    { return a.v[k+a.offset] }
func (a *vArr) set(k int, x int) { a.v[k+a.offset] = x }

func maxD(len1, len2 int) int { return (len1+len2+1)/2 + 1 }

func commonPrefixLen(old []string, oldStart, oldEnd int, new []string, newStart, newEnd int) int {
	if oldStart >= oldEnd || newStart >= newEnd {
		return 0
	}
	count := 0
	for i, j := newStart, oldStart; i < newEnd && j < oldEnd && new[i] == old[j]; i, j = i+1, j+1 {
		count++
	}
	return count
}

func commonSuffixLen(old []string, oldStart, oldEnd int, new []string, newStart, newEnd int) int {
	if oldStart >= oldEnd || newStart >= newEnd {
		return 0
	}
	count := 0
	for i, j := newEnd-1, oldEnd-1; i >= newStart && j >= oldStart && new[i] == old[j]; i, j = i-1, j-1 {
		count++
	}
	return count
}

// findMiddleSnake is the divide step: run the basic algorithm forward and
// backward until the furthest-reaching paths overlap.
func (m *myersState) findMiddleSnake(oldStart, oldEnd, newStart, newEnd int) (int, int, bool) {
	n := oldEnd - oldStart
	mm := newEnd - newStart

	// By Lemma 1 in the paper, the optimal edit script length is odd or even
	// as delta is odd or even.
	delta := n - mm
	odd := delta&1 == 1

	vf, vb := m.vf, m.vb
	vf.set(1, 0)
	vb.set(1, 0)

	dMax := maxD(n, mm)
	for d := 0; d < dMax; d++ {
		// Forward path.
		for k := d; k >= -d; k -= 2 {
			var x int
			if k == -d || (k != d && vf.get(k-1) < vf.get(k+1)) {
				x = vf.get(k + 1)
			} else {
				x = vf.get(k-1) + 1
			}
			y := x - k

			// Snake start; extend through the graph at no cost.
			x0, y0 := x, y
			if x < n && y < mm {
				x += commonPrefixLen(m.old, oldStart+x, oldEnd, m.new, newStart+y, newEnd)
			}
			vf.set(k, x)

			if odd && abs(k-delta) <= d-1 {
				if vf.get(k)+vb.get(-(k-delta)) >= n {
					return x0 + oldStart, y0 + newStart, true
				}
			}
		}

		// Backward path.
		for k := d; k >= -d; k -= 2 {
			var x int
			if k == -d || (k != d && vb.get(k-1) < vb.get(k+1)) {
				x = vb.get(k + 1)
			} else {
				x = vb.get(k-1) + 1
			}
			y := x - k

			if x < n && y < mm {
				advance := commonSuffixLen(m.old, oldStart, oldStart+n-x, m.new, newStart, newStart+mm-y)
				x += advance
				y += advance
			}
			vb.set(k, x)

			if !odd && abs(k-delta) <= d {
				if vb.get(k)+vf.get(-(k-delta)) >= n {
					return n - x + oldStart, mm - y + newStart, true
				}
			}
		}
	}
	return 0, 0, false
}

func (m *myersState) conquer(oldStart, oldEnd, newStart, newEnd int) {
	prefixLen := commonPrefixLen(m.old, oldStart, oldEnd, m.new, newStart, newEnd)
	if prefixLen > 0 {
		m.equal(oldStart, newStart, prefixLen)
	}
	oldStart += prefixLen
	newStart += prefixLen

	suffixLen := commonSuffixLen(m.old, oldStart, oldEnd, m.new, newStart, newEnd)
	suffixOld := oldEnd - suffixLen
	suffixNew := newEnd - suffixLen
	oldEnd -= suffixLen
	newEnd -= suffixLen

	oldEmpty := oldStart >= oldEnd
	newEmpty := newStart >= newEnd
	switch {
	case oldEmpty && newEmpty:
		// Nothing.
	case newEmpty:
		m.delete(oldStart, oldEnd-oldStart, newStart)
	case oldEmpty:
		m.insert(oldStart, newStart, newEnd-newStart)
	default:
		if xStart, yStart, ok := m.findMiddleSnake(oldStart, oldEnd, newStart, newEnd); ok {
			m.conquer(oldStart, xStart, newStart, yStart)
			m.conquer(xStart, oldEnd, yStart, newEnd)
		} else {
			// Unreachable without a deadline; kept for structural parity.
			m.delete(oldStart, oldEnd-oldStart, newStart)
			m.insert(oldStart, newStart, newEnd-newStart)
		}
	}

	if suffixLen > 0 {
		m.equal(suffixOld, suffixNew, suffixLen)
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// cleanupDiffOps is similar's Compact pass: walks through all edits and shifts
// them up and then down, merging similar edits where they connect.
func cleanupDiffOps(old, new []string, ops *[]rangeOp) {
	// First attempt to compact all Deletions.
	pointer := 0
	for pointer < len(*ops) {
		if (*ops)[pointer].tag == tagDelete {
			pointer = shiftDiffOpsUp(ops, old, new, pointer)
			pointer = shiftDiffOpsDown(ops, old, new, pointer)
		}
		pointer++
	}
	// Then attempt to compact all Insertions.
	pointer = 0
	for pointer < len(*ops) {
		if (*ops)[pointer].tag == tagInsert {
			pointer = shiftDiffOpsUp(ops, old, new, pointer)
			pointer = shiftDiffOpsDown(ops, old, new, pointer)
		}
		pointer++
	}
}

func removeOp(ops *[]rangeOp, i int) { *ops = append((*ops)[:i], (*ops)[i+1:]...) }

func insertOp(ops *[]rangeOp, i int, op rangeOp) {
	*ops = append(*ops, rangeOp{})
	copy((*ops)[i+1:], (*ops)[i:])
	(*ops)[i] = op
}

func shiftDiffOpsUp(ops *[]rangeOp, old, new []string, pointer int) int {
	for pointer > 0 {
		prevOp := (*ops)[pointer-1]
		thisOp := (*ops)[pointer]
		switch {
		// Shift Inserts upwards.
		case thisOp.tag == tagInsert && prevOp.tag == tagEqual:
			po1, po2 := prevOp.oldRange()
			tn1, tn2 := thisOp.newRange()
			suffixLen := commonSuffixLen(old, po1, po2, new, tn1, tn2)
			if suffixLen > 0 {
				if pointer+1 < len(*ops) && (*ops)[pointer+1].tag == tagEqual {
					(*ops)[pointer+1].growLeft(suffixLen)
				} else {
					insertOp(ops, pointer+1, rangeOp{
						tag:      tagEqual,
						oldIndex: po2 - suffixLen,
						oldLen:   suffixLen,
						newIndex: tn2 - suffixLen,
						newLen:   suffixLen,
					})
				}
				(*ops)[pointer].shiftLeft(suffixLen)
				(*ops)[pointer-1].shrinkLeft(suffixLen)
				if (*ops)[pointer-1].isEmpty() {
					removeOp(ops, pointer-1)
					pointer--
				}
			} else if (*ops)[pointer-1].isEmpty() {
				removeOp(ops, pointer-1)
				pointer--
			} else {
				return pointer // can't shift upwards anymore
			}
		// Shift Deletions upwards.
		case thisOp.tag == tagDelete && prevOp.tag == tagEqual:
			po1, po2 := prevOp.oldRange()
			tn1, tn2 := thisOp.newRange()
			suffixLen := commonSuffixLen(old, po1, po2, new, tn1, tn2)
			if suffixLen != 0 {
				if pointer+1 < len(*ops) && (*ops)[pointer+1].tag == tagEqual {
					(*ops)[pointer+1].growLeft(suffixLen)
				} else {
					// NB: len here mirrors similar 2.7.0 exactly (it differs
					// from the Insert branch); parity over aesthetics.
					insertOp(ops, pointer+1, rangeOp{
						tag:      tagEqual,
						oldIndex: po2 - suffixLen,
						oldLen:   (po2 - po1) - suffixLen,
						newIndex: tn2 - suffixLen,
						newLen:   (po2 - po1) - suffixLen,
					})
				}
				(*ops)[pointer].shiftLeft(suffixLen)
				(*ops)[pointer-1].shrinkLeft(suffixLen)
				if (*ops)[pointer-1].isEmpty() {
					removeOp(ops, pointer-1)
					pointer--
				}
			} else if (*ops)[pointer-1].isEmpty() {
				removeOp(ops, pointer-1)
				pointer--
			} else {
				return pointer
			}
		// Swap the Delete and Insert.
		case (thisOp.tag == tagInsert && prevOp.tag == tagDelete) ||
			(thisOp.tag == tagDelete && prevOp.tag == tagInsert):
			(*ops)[pointer-1], (*ops)[pointer] = (*ops)[pointer], (*ops)[pointer-1]
			pointer--
		// Merge the two ranges.
		case thisOp.tag == tagInsert && prevOp.tag == tagInsert:
			(*ops)[pointer-1].growRight(thisOp.newLen)
			removeOp(ops, pointer)
			pointer--
		case thisOp.tag == tagDelete && prevOp.tag == tagDelete:
			(*ops)[pointer-1].growRight(thisOp.oldLen)
			removeOp(ops, pointer)
			pointer--
		default:
			panic("unexpected tag")
		}
	}
	return pointer
}

func shiftDiffOpsDown(ops *[]rangeOp, old, new []string, pointer int) int {
	for pointer+1 < len(*ops) {
		nextOp := (*ops)[pointer+1]
		thisOp := (*ops)[pointer]
		switch {
		// Shift Inserts downwards.
		case thisOp.tag == tagInsert && nextOp.tag == tagEqual:
			no1, no2 := nextOp.oldRange()
			tn1, tn2 := thisOp.newRange()
			prefixLen := commonPrefixLen(old, no1, no2, new, tn1, tn2)
			if prefixLen > 0 {
				if pointer > 0 && (*ops)[pointer-1].tag == tagEqual {
					(*ops)[pointer-1].growRight(prefixLen)
				} else {
					insertOp(ops, pointer, rangeOp{
						tag:      tagEqual,
						oldIndex: no1,
						oldLen:   prefixLen,
						newIndex: tn1,
						newLen:   prefixLen,
					})
					pointer++
				}
				(*ops)[pointer].shiftRight(prefixLen)
				(*ops)[pointer+1].shrinkRight(prefixLen)
				if (*ops)[pointer+1].isEmpty() {
					removeOp(ops, pointer+1)
				}
			} else if (*ops)[pointer+1].isEmpty() {
				removeOp(ops, pointer+1)
			} else {
				return pointer // can't shift downwards anymore
			}
		// Shift Deletions downwards.
		case thisOp.tag == tagDelete && nextOp.tag == tagEqual:
			no1, no2 := nextOp.oldRange()
			tn1, tn2 := thisOp.newRange()
			prefixLen := commonPrefixLen(old, no1, no2, new, tn1, tn2)
			if prefixLen > 0 {
				if pointer > 0 && (*ops)[pointer-1].tag == tagEqual {
					(*ops)[pointer-1].growRight(prefixLen)
				} else {
					insertOp(ops, pointer, rangeOp{
						tag:      tagEqual,
						oldIndex: no1,
						oldLen:   prefixLen,
						newIndex: tn1,
						newLen:   prefixLen,
					})
					pointer++
				}
				(*ops)[pointer].shiftRight(prefixLen)
				(*ops)[pointer+1].shrinkRight(prefixLen)
				if (*ops)[pointer+1].isEmpty() {
					removeOp(ops, pointer+1)
				}
			} else if (*ops)[pointer+1].isEmpty() {
				removeOp(ops, pointer+1)
			} else {
				return pointer
			}
		// Swap the Delete and Insert.
		case (thisOp.tag == tagInsert && nextOp.tag == tagDelete) ||
			(thisOp.tag == tagDelete && nextOp.tag == tagInsert):
			(*ops)[pointer], (*ops)[pointer+1] = (*ops)[pointer+1], (*ops)[pointer]
			pointer++
		// Merge the two ranges.
		case thisOp.tag == tagInsert && nextOp.tag == tagInsert:
			(*ops)[pointer].growRight(nextOp.newLen)
			removeOp(ops, pointer+1)
		case thisOp.tag == tagDelete && nextOp.tag == tagDelete:
			(*ops)[pointer].growRight(nextOp.oldLen)
			removeOp(ops, pointer+1)
		default:
			panic("unexpected tag")
		}
	}
	return pointer
}
