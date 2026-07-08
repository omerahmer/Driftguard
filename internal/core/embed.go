package core

// Embedding providers (port of embed.rs).
//
// All selection logic depends on the Embedder interface, not on Voyage
// directly: it keeps the similarity/selection code testable without a network
// call or an API key, and swapping the provider later is a one-impl change.
//
// Why Voyage at all: Anthropic does not offer a first-party embeddings API and
// recommends Voyage AI as its embeddings partner. The Anthropic key is still
// used — for the LLM judge — just not here.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
)

// InputType is Voyage's asymmetric retrieval hint. Embedding the prompt's
// changed span as a Query and each eval case's expected_behavior as a Document
// matches the retrieval framing ("which described behaviors does this change
// retrieve?") and tends to improve relevance over embedding both sides
// identically.
type InputType string

const (
	InputQuery    InputType = "query"
	InputDocument InputType = "document"
)

// Embedder is a source of text embeddings.
type Embedder interface {
	// Dimension is the output dimension — must match the pgvector column.
	Dimension() int
	// Embed a batch of texts, returning one vector per input in input order.
	Embed(ctx context.Context, inputs []string, inputType InputType) ([][]float32, error)
}

const (
	voyageDefaultModel     = "voyage-3.5-lite"
	voyageDefaultDimension = 1024
	voyageEndpoint         = "https://api.voyageai.com/v1/embeddings"
)

// VoyageEmbedder is Voyage AI embeddings (voyage-3.5-lite, 1024-dim).
type VoyageEmbedder struct {
	client    *http.Client
	apiKey    string
	model     string
	dimension int
}

// NewVoyageEmbedder constructs from VOYAGE_API_KEY. Returns a clear error so
// the CLI can print a clean message when the key is missing — an empty key
// would otherwise reach Voyage and come back as an opaque 401.
func NewVoyageEmbedder() (*VoyageEmbedder, error) {
	apiKey := strings.TrimSpace(os.Getenv("VOYAGE_API_KEY"))
	if apiKey == "" {
		return nil, fmt.Errorf("VOYAGE_API_KEY is not set — needed for embeddings in `select-evals`. " +
			"Get a key at voyageai.com and add it to .env.")
	}
	// Model is configurable via VOYAGE_MODEL. The DIMENSION is intentionally
	// NOT env-configurable: it must match the pgvector column (vector(1024) in
	// migration 0002), so changing it requires a migration.
	model := strings.TrimSpace(os.Getenv("VOYAGE_MODEL"))
	if model == "" {
		model = voyageDefaultModel
	}
	return &VoyageEmbedder{
		client:    &http.Client{},
		apiKey:    apiKey,
		model:     model,
		dimension: voyageDefaultDimension,
	}, nil
}

func (v *VoyageEmbedder) Dimension() int { return v.dimension }

func (v *VoyageEmbedder) Embed(ctx context.Context, inputs []string, inputType InputType) ([][]float32, error) {
	if len(inputs) == 0 {
		return nil, nil
	}

	body, err := json.Marshal(map[string]any{
		"input":            inputs,
		"model":            v.model,
		"input_type":       string(inputType),
		"output_dimension": v.dimension,
	})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, voyageEndpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+v.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := v.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		detail, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedding provider error: Voyage API returned %s: %s", resp.Status, detail)
	}

	var parsed struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, err
	}
	// The API documents in-order results, but sort by index defensively so a
	// reordering never silently misaligns embeddings with their inputs.
	sort.Slice(parsed.Data, func(i, j int) bool { return parsed.Data[i].Index < parsed.Data[j].Index })
	out := make([][]float32, len(parsed.Data))
	for i, item := range parsed.Data {
		out[i] = item.Embedding
	}
	return out, nil
}

// ToPgvectorLiteral serializes a vector to pgvector's text input format, e.g.
// `[0.1,0.2,0.3]`. We bind embeddings as this text literal and cast with
// `$1::vector` in SQL rather than pulling in a pgvector binding library —
// the wire format stays something we fully control and can debug by eye.
// 'f'/-1/32 renders the shortest fixed-notation string that round-trips the
// float32, matching Rust's f32::to_string.
func ToPgvectorLiteral(vector []float32) string {
	var b strings.Builder
	b.Grow(len(vector)*8 + 2)
	b.WriteByte('[')
	for i, val := range vector {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatFloat(float64(val), 'f', -1, 32))
	}
	b.WriteByte(']')
	return b.String()
}
