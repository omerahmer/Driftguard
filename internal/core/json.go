package core

import (
	"bytes"
	"encoding/json"
)

// marshalCompact serializes like Rust's serde_json::to_string: compact and
// WITHOUT Go's default HTML escaping (serde doesn't escape <, >, &). Used for
// values whose serialized form is persisted (diff_from_parent) or compared
// against Rust output, where byte-compatibility matters.
func marshalCompact(v any) (string, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return "", err
	}
	return string(bytes.TrimRight(buf.Bytes(), "\n")), nil
}
