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
	"strings"

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

	mux.HandleFunc("GET /prompts/{id}/eval-cases", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid uuid")
			return
		}
		respond(w, func(ctx context.Context) (any, error) {
			return core.ListEvalCases(ctx, pool, id)
		}, r)
	})

	// The one write endpoint: create an eval case. Prompts still originate from
	// the CLI/scan (the codebase is their source of truth); eval cases have no
	// such home, so authoring them here is the scoped write the dashboard needs.
	mux.HandleFunc("POST /prompts/{id}/eval-cases", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid uuid")
			return
		}
		var body struct {
			Input            string `json:"input"`
			ExpectedBehavior string `json:"expected_behavior"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON body")
			return
		}
		if strings.TrimSpace(body.Input) == "" || strings.TrimSpace(body.ExpectedBehavior) == "" {
			writeError(w, http.StatusBadRequest, "input and expected_behavior are required")
			return
		}
		ctx := r.Context()
		prompt, err := core.GetPromptByID(ctx, pool, id)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		if prompt == nil {
			writeError(w, http.StatusNotFound, "no prompt with that id")
			return
		}
		created, err := core.CreateEvalCaseForPrompt(ctx, pool, id, body.Input, body.ExpectedBehavior)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(created)
	})

	// Behavior diffs to hand-label, and the judge's audit against those labels.
	mux.HandleFunc("GET /behavior-diffs", func(w http.ResponseWriter, r *http.Request) {
		onlyUnlabeled := r.URL.Query().Get("state") == "unlabeled"
		respond(w, func(ctx context.Context) (any, error) {
			return core.ListBehaviorDiffs(ctx, pool, onlyUnlabeled)
		}, r)
	})

	mux.HandleFunc("GET /judge-audit", func(w http.ResponseWriter, r *http.Request) {
		respond(w, func(ctx context.Context) (any, error) {
			return core.ComputeJudgeAudit(ctx, pool)
		}, r)
	})

	// Record an independent human label (the gold ground truth).
	mux.HandleFunc("POST /behavior-diffs/{id}/label", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid uuid")
			return
		}
		var body struct {
			Changed *bool `json:"changed"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Changed == nil {
			writeError(w, http.StatusBadRequest, "body must be {\"changed\": true|false}")
			return
		}
		if err := core.SetHumanLabel(r.Context(), pool, id, *body.Changed); err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		w.WriteHeader(http.StatusNoContent)
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
		// ?ground_truth=human scores against hand labels; default judge.
		gt := core.GroundTruth(r.URL.Query().Get("ground_truth"))
		respond(w, func(ctx context.Context) (any, error) {
			return core.Score(ctx, pool, prompt, core.DefaultThresholds(), gt)
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
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
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
