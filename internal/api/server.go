// Package api is the read-only HTTP API for the dashboard.
//
// Every handler calls a core read function (the same ones the CLI uses) and
// serializes the result. There is deliberately no domain logic and no write
// path here — writes go through the CLI/Action, and this is a thin second
// consumer of core. Precision/recall is computed live per request (derived
// state), not read from a stored column.
//
// Routing is stdlib net/http (Go 1.22+ method/path patterns) — the endpoint
// surface is small enough that a router dependency buys nothing.
package api

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/omerahmer/driftguard/internal/core"
)

// NewServer builds the API handler over a shared pool.
func NewServer(pool *pgxpool.Pool) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("ok"))
	})

	mux.HandleFunc("GET /prompts", func(w http.ResponseWriter, r *http.Request) {
		respond(w, func(ctx context.Context) (any, error) {
			return core.ListPrompts(ctx, pool)
		}, r)
	})

	mux.HandleFunc("GET /prompts/{id}/versions", func(w http.ResponseWriter, r *http.Request) {
		// Each version carries diff_from_parent (the PromptDiff JSON) — that's
		// the diff viewer's data, so no separate diff endpoint is needed.
		id, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid uuid")
			return
		}
		respond(w, func(ctx context.Context) (any, error) {
			return core.ListVersions(ctx, pool, id)
		}, r)
	})

	mux.HandleFunc("GET /versions/{id}/runs", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid uuid")
			return
		}
		respond(w, func(ctx context.Context) (any, error) {
			return core.ListEvalRuns(ctx, pool, id)
		}, r)
	})

	mux.HandleFunc("GET /score", func(w http.ResponseWriter, r *http.Request) {
		var prompt *string
		if p := r.URL.Query().Get("prompt"); p != "" {
			prompt = &p
		}
		respond(w, func(ctx context.Context) (any, error) {
			return core.Score(ctx, pool, prompt, core.DefaultThresholds())
		}, r)
	})

	// Permissive CORS is fine for a local read-only dashboard; the Next.js dev
	// proxy also makes it same-origin. Tighten if ever deployed.
	return logRequests(cors(mux))
}

func respond(w http.ResponseWriter, fn func(context.Context) (any, error), r *http.Request) {
	result, err := fn(r.Context())
	if err != nil {
		status := http.StatusInternalServerError
		// Not-found variants map to 404; everything else is a 500.
		var pnf core.PromptNotFoundError
		var vnf core.VersionNotFoundError
		if errors.As(err, &pnf) || errors.As(err, &vnf) {
			status = http.StatusNotFound
		}
		writeError(w, status, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(result); err != nil {
		log.Printf("encode response: %v", err)
	}
}

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "*")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		next.ServeHTTP(w, r)
		log.Printf("%s %s", r.Method, r.URL.Path)
	})
}
