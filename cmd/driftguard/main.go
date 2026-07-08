// driftguard — version LLM prompts and predict which evals a change affects.
//
// Go rewrite of the Rust CLI. Subcommands are thin delivery layers over
// internal/core; they own flag parsing, printing, and exit codes, nothing else.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/joho/godotenv"
	"github.com/spf13/cobra"

	"github.com/omerahmer/driftguard/internal/api"
	"github.com/omerahmer/driftguard/internal/core"
)

func main() {
	// .env is optional (CI sets real env vars); ignore a missing file.
	godotenv.Load()

	root := &cobra.Command{
		Use:           "driftguard",
		Short:         "Version LLM prompts and predict which evals a change affects.",
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	root.AddCommand(promptCmd(), apiCmd())

	if err := root.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// connect opens the pool from DATABASE_URL (same contract as the Rust CLI).
func connect(ctx context.Context) (*pgxpool.Pool, error) {
	url := os.Getenv("DATABASE_URL")
	if url == "" {
		return nil, fmt.Errorf("DATABASE_URL is not set")
	}
	return core.Connect(ctx, url)
}

func promptCmd() *cobra.Command {
	cmd := &cobra.Command{Use: "prompt", Short: "Manage prompts and their versions"}

	var asJSON bool
	list := &cobra.Command{
		Use:   "list",
		Short: "List registered prompts",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			pool, err := connect(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()

			prompts, err := core.ListPrompts(ctx, pool)
			if err != nil {
				return err
			}
			if asJSON {
				return json.NewEncoder(os.Stdout).Encode(prompts)
			}
			if len(prompts) == 0 {
				fmt.Println("no prompts registered")
				return nil
			}
			for _, p := range prompts {
				current := "-"
				if p.CurrentVersionID != nil {
					current = p.CurrentVersionID.String()[:8]
				}
				fmt.Printf("%s  %-20s  current=%s  created=%s\n",
					p.ID.String()[:8], p.Name, current, p.CreatedAt.Format("2006-01-02"))
			}
			return nil
		},
	}
	list.Flags().BoolVar(&asJSON, "json", false, "machine-readable output")

	create := &cobra.Command{
		Use:   "create <name>",
		Short: "Register a prompt (the first `prompt version` becomes v1)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			pool, err := connect(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()
			p, err := core.CreatePrompt(ctx, pool, args[0])
			if err != nil {
				return err
			}
			fmt.Printf("created prompt %q (%s)\n", p.Name, p.ID)
			return nil
		},
	}

	var contentFile string
	var versionJSON bool
	version := &cobra.Command{
		Use:   "version <name>",
		Short: "Add a new version of a prompt (content from --file or stdin)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			var raw []byte
			var err error
			if contentFile != "" {
				raw, err = os.ReadFile(contentFile)
			} else {
				raw, err = io.ReadAll(cmd.InOrStdin())
			}
			if err != nil {
				return err
			}

			ctx := cmd.Context()
			pool, err := connect(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()
			out, err := core.CreateVersion(ctx, pool, args[0], string(raw))
			if err != nil {
				return err
			}
			if versionJSON {
				return json.NewEncoder(os.Stdout).Encode(out)
			}
			if out.Diff == nil {
				fmt.Printf("created v1 %s (no parent)\n", out.Version.ID)
			} else if out.Unchanged {
				fmt.Printf("created %s — content identical to parent (empty changed span)\n", out.Version.ID)
			} else {
				fmt.Printf("created %s: +%d -%d =%d segments\n", out.Version.ID,
					out.Diff.Stats.Added, out.Diff.Stats.Removed, out.Diff.Stats.Unchanged)
				for _, s := range out.Diff.ChangedSpans {
					fmt.Printf("  + %s\n", s)
				}
				for _, s := range out.Diff.RemovedSpans {
					fmt.Printf("  - %s\n", s)
				}
			}
			return nil
		},
	}
	version.Flags().StringVar(&contentFile, "file", "", "read content from a file instead of stdin")
	version.Flags().BoolVar(&versionJSON, "json", false, "machine-readable output")

	cmd.AddCommand(list, create, version)
	return cmd
}

func apiCmd() *cobra.Command {
	var addr string
	cmd := &cobra.Command{
		Use:   "api",
		Short: "Serve the read-only HTTP API for the dashboard",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			pool, err := connect(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()

			fmt.Printf("driftguard-api listening on http://%s\n", addr)
			return http.ListenAndServe(addr, api.NewServer(pool))
		},
	}
	cmd.Flags().StringVar(&addr, "addr", "0.0.0.0:3000", "listen address")
	return cmd
}
