# Driftguard convenience targets (single Go binary since the Go rewrite).

.PHONY: build migrate test api dashboard clean

build:
	go build -o driftguard ./cmd/driftguard

migrate:
	sqlx migrate run

# Includes the DB-backed parity/round-trip tests when DATABASE_URL is set
# (they self-skip otherwise).
test:
	go test ./...

# Read API (Phase 6) on :3000.
api:
	go run ./cmd/driftguard api

# Next.js dashboard (Phase 6) on :3001 — proxies /api to the API above.
dashboard:
	cd dashboard && npm run dev

clean:
	rm -f driftguard
