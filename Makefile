# Driftguard convenience targets.

SIDECAR := tools/driftguard-llm/driftguard-llm

.PHONY: sidecar build migrate verify test clean

# Build the Go LLM sidecar (uses the official anthropic-sdk-go).
sidecar:
	cd tools/driftguard-llm && go build -o driftguard-llm .

# Build everything: the Rust workspace + the Go sidecar.
build: sidecar
	cargo build

migrate:
	sqlx migrate run

# Offline pipeline checks (no API keys needed).
verify:
	cargo run -p driftguard-core --example verify_pgvector
	cargo run -p driftguard-core --example verify_selection
	cargo run -p driftguard-core --example verify_validation

test:
	cargo test --workspace

clean:
	cargo clean
	rm -f $(SIDECAR)
