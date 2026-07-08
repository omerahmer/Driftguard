// driftguard — version LLM prompts and predict which evals a change affects.
//
// Go rewrite of the Rust CLI. Subcommands are thin delivery layers over
// internal/core; they own flag parsing, printing, and exit codes, nothing else.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

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
	// Single root-level --json accepted after any subcommand, mirroring the
	// Rust CLI's global flag.
	root.PersistentFlags().BoolVar(&jsonOut, "json", false, "machine-readable output")
	root.AddCommand(promptCmd(), evalCmd(), fixturesCmd(), selectEvalsCmd(),
		runEvalsCmd(), validateCmd(), judgeValidateCmd(), scoreCmd(), ciCmd(), apiCmd())

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
			if jsonOut {
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
			if jsonOut {
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

// jsonOut is the root-level --json flag (global, like the Rust CLI's).
var jsonOut bool

func emit(v any) error { return json.NewEncoder(os.Stdout).Encode(v) }

func evalCmd() *cobra.Command {
	cmd := &cobra.Command{Use: "eval", Short: "Manage eval cases"}

	var promptName, input, expected string
	add := &cobra.Command{
		Use:   "add",
		Short: "Add an eval case to a prompt",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			pool, err := connect(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()
			c, err := core.CreateEvalCase(ctx, pool, promptName, input, expected)
			if err != nil {
				return err
			}
			if jsonOut {
				return emit(c)
			}
			fmt.Printf("added eval case %s to %q\n", c.ID, promptName)
			return nil
		},
	}
	add.Flags().StringVar(&promptName, "prompt", "", "prompt name")
	add.Flags().StringVar(&input, "input", "", "test input")
	add.Flags().StringVar(&expected, "expected", "", "expected behavior description")
	add.MarkFlagRequired("prompt")
	add.MarkFlagRequired("input")
	add.MarkFlagRequired("expected")

	var listPrompt string
	list := &cobra.Command{
		Use:   "list",
		Short: "List a prompt's eval cases",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			pool, err := connect(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()
			p, err := core.GetPromptByName(ctx, pool, listPrompt)
			if err != nil {
				return err
			}
			if p == nil {
				return core.PromptNotFoundError{Name: listPrompt}
			}
			cases, err := core.ListEvalCases(ctx, pool, p.ID)
			if err != nil {
				return err
			}
			if jsonOut {
				return emit(cases)
			}
			for _, c := range cases {
				fmt.Printf("%s  input=%q\n          expected=%q\n", c.ID.String()[:8], c.Input, c.ExpectedBehavior)
			}
			return nil
		},
	}
	list.Flags().StringVar(&listPrompt, "prompt", "", "prompt name")
	list.MarkFlagRequired("prompt")

	cmd.AddCommand(add, list)
	return cmd
}

func fixturesCmd() *cobra.Command {
	cmd := &cobra.Command{Use: "fixtures", Short: "Load fixture prompts/evals/edits"}

	var file string
	var force bool
	load := &cobra.Command{
		Use:   "load",
		Short: "Load a fixtures TOML into the registry",
		RunE: func(cmd *cobra.Command, _ []string) error {
			raw, err := os.ReadFile(file)
			if err != nil {
				return err
			}
			fx, err := core.ParseFixtures(string(raw))
			if err != nil {
				return err
			}
			ctx := cmd.Context()
			pool, err := connect(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()
			s, err := core.LoadFixtures(ctx, pool, fx, force)
			if err != nil {
				return err
			}
			if jsonOut {
				return emit(s)
			}
			fmt.Printf("Loaded %d prompt(s), %d eval case(s), %d version(s).", s.Prompts, s.EvalCases, s.Versions)
			if s.SkippedExisting > 0 {
				fmt.Printf(" Skipped %d already-loaded prompt(s) (use --force to reload).", s.SkippedExisting)
			}
			fmt.Println()
			return nil
		},
	}
	load.Flags().StringVar(&file, "file", "", "fixtures TOML path")
	load.Flags().BoolVar(&force, "force", false, "delete and recreate already-loaded prompts")
	load.MarkFlagRequired("file")

	cmd.AddCommand(load)
	return cmd
}

func selectEvalsCmd() *cobra.Command {
	var promptName, versionRef string
	var threshold float64
	cmd := &cobra.Command{
		Use:   "select-evals",
		Short: "Predict which eval cases a version's change affects",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			pool, err := connect(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()
			embedder, err := core.NewVoyageEmbedder()
			if err != nil {
				return err
			}
			out, err := core.SelectEvals(ctx, pool, embedder, promptName, versionRef, threshold)
			if err != nil {
				return err
			}
			if jsonOut {
				return emit(out)
			}
			fmt.Printf("%q %s: %d of %d selected (threshold %.2f, %.0f%% reduction)\n",
				out.Prompt, out.VersionID, out.Selected, out.Total, out.Threshold, out.Reduction*100)
			for _, it := range out.Items {
				mark := " "
				if it.WasSelected {
					mark = "*"
				}
				fmt.Printf("  %s %.3f  %s\n", mark, it.Similarity, it.ExpectedBehavior)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&promptName, "prompt", "", "prompt name")
	cmd.Flags().StringVar(&versionRef, "version", "latest", "version ref (latest/parent/vN/id)")
	cmd.Flags().Float64Var(&threshold, "threshold", 0.5, "similarity threshold")
	cmd.MarkFlagRequired("prompt")
	return cmd
}

func runEvalsCmd() *cobra.Command {
	var promptName, versionRef string
	var concurrency int
	cmd := &cobra.Command{
		Use:   "run-evals",
		Short: "Generate outputs for every eval case against one version",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			pool, err := connect(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()
			llm, err := core.NewAnthropicLlm()
			if err != nil {
				return err
			}
			n, err := core.RunEvals(ctx, pool, llm, promptName, versionRef, concurrency)
			if err != nil {
				return err
			}
			fmt.Printf("ran %d eval case(s) against %q %s\n", n, promptName, versionRef)
			return nil
		},
	}
	cmd.Flags().StringVar(&promptName, "prompt", "", "prompt name")
	cmd.Flags().StringVar(&versionRef, "version", "latest", "version ref")
	cmd.Flags().IntVar(&concurrency, "concurrency", 6, "in-flight LLM calls")
	cmd.MarkFlagRequired("prompt")
	return cmd
}

func validateCmd() *cobra.Command {
	var fixturesPath string
	var threshold float64
	var concurrency int
	cmd := &cobra.Command{
		Use:   "validate",
		Short: "Run the full fixtures validation pipeline (evals + judge + ground truth)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			raw, err := os.ReadFile(fixturesPath)
			if err != nil {
				return err
			}
			fx, err := core.ParseFixtures(string(raw))
			if err != nil {
				return err
			}
			ctx := cmd.Context()
			pool, err := connect(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()
			llm, err := core.NewAnthropicLlm()
			if err != nil {
				return err
			}
			embedder, err := core.NewVoyageEmbedder()
			if err != nil {
				return err
			}
			// Loading first is idempotent (existing prompts are skipped).
			if _, err := core.LoadFixtures(ctx, pool, fx, false); err != nil {
				return err
			}
			s, err := core.ValidateFixtures(ctx, pool, llm, embedder, fx, threshold, concurrency)
			if err != nil {
				return err
			}
			if jsonOut {
				return emit(s)
			}
			fmt.Printf("validated %d prompt(s), %d edit(s): %d versions evaluated, %d eval runs, %d behavior diffs\n",
				s.Prompts, s.Edits, s.VersionsEvaluated, s.EvalRuns, s.BehaviorDiffs)
			return nil
		},
	}
	cmd.Flags().StringVar(&fixturesPath, "fixtures", "", "fixtures TOML path")
	cmd.Flags().Float64Var(&threshold, "threshold", 0.5, "similarity threshold")
	cmd.Flags().IntVar(&concurrency, "concurrency", 6, "in-flight LLM calls")
	cmd.MarkFlagRequired("fixtures")
	return cmd
}

func judgeValidateCmd() *cobra.Command {
	var sampleSize int64
	cmd := &cobra.Command{
		Use:   "judge-validate",
		Short: "Human spot-check of judge verdicts; reports agreement rate",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			pool, err := connect(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()

			samples, err := core.SampleUnvalidatedDiffs(ctx, pool, sampleSize)
			if err != nil {
				return err
			}
			reader := bufio.NewReader(cmd.InOrStdin())
			for i, s := range samples {
				verdict := "?"
				if s.JudgeBehaviorChanged != nil {
					verdict = fmt.Sprintf("%v", *s.JudgeBehaviorChanged)
				}
				fmt.Printf("\n[%d/%d] input: %s\nexpected: %s\n--- before ---\n%s\n--- after ---\n%s\njudge says behavior_changed=%s\n",
					i+1, len(samples), s.Input, s.ExpectedBehavior, s.BeforeOutput, s.AfterOutput, verdict)
				fmt.Print("agree? [y/n/s(kip)] ")
				line, _ := reader.ReadString('\n')
				switch strings.TrimSpace(strings.ToLower(line)) {
				case "y":
					if err := core.SetHumanAgreed(ctx, pool, s.BehaviorDiffID, true); err != nil {
						return err
					}
				case "n":
					if err := core.SetHumanAgreed(ctx, pool, s.BehaviorDiffID, false); err != nil {
						return err
					}
				}
			}
			agreed, total, err := core.JudgeAgreement(ctx, pool)
			if err != nil {
				return err
			}
			if total > 0 {
				fmt.Printf("\njudge agreement: %d/%d (%.0f%%)\n", agreed, total, float64(agreed)/float64(total)*100)
			} else {
				fmt.Println("\nno human-reviewed verdicts yet")
			}
			return nil
		},
	}
	cmd.Flags().Int64Var(&sampleSize, "sample-size", 10, "verdicts to sample")
	return cmd
}

func scoreCmd() *cobra.Command {
	var promptName string
	cmd := &cobra.Command{
		Use:   "score",
		Short: "Precision/recall/reduction across the threshold sweep",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			pool, err := connect(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()
			var filter *string
			if promptName != "" {
				filter = &promptName
			}
			report, err := core.Score(ctx, pool, filter, core.DefaultThresholds())
			if err != nil {
				return err
			}
			if jsonOut {
				return emit(report)
			}
			fmt.Printf("samples: %d\n", report.Samples)
			fmt.Println("thresh  reduction  behavior P / R      flip P / R")
			for _, r := range report.Rows {
				fmt.Printf("%.2f    %5.1f%%     %.3f / %.3f     %.3f / %.3f\n",
					r.Threshold, r.Reduction*100,
					r.Behavior.Precision, r.Behavior.Recall,
					r.Flip.Precision, r.Flip.Recall)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&promptName, "prompt", "", "filter to one prompt")
	return cmd
}

func ciCmd() *cobra.Command {
	var basePath, headPath string
	var threshold float64
	var warnOnly bool
	var concurrency int
	cmd := &cobra.Command{
		Use:   "ci",
		Short: "PR gate: select → run selected → judge over a base/head fixtures pair",
		RunE: func(cmd *cobra.Command, _ []string) error {
			baseRaw, err := os.ReadFile(basePath)
			if err != nil {
				return err
			}
			headRaw, err := os.ReadFile(headPath)
			if err != nil {
				return err
			}
			base, err := core.ParseFixtures(string(baseRaw))
			if err != nil {
				return err
			}
			head, err := core.ParseFixtures(string(headRaw))
			if err != nil {
				return err
			}

			ctx := cmd.Context()
			pool, err := connect(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()
			llm, err := core.NewAnthropicLlm()
			if err != nil {
				return err
			}
			embedder, err := core.NewVoyageEmbedder()
			if err != nil {
				return err
			}

			report, err := core.RunCi(ctx, pool, llm, embedder, base, head, threshold, concurrency)
			if err != nil {
				return err
			}
			if jsonOut {
				if err := emit(report); err != nil {
					return err
				}
			} else {
				// stdout is the PR comment body (the workflow redirects it).
				fmt.Print(core.RenderComment(report))
			}
			if report.Failed > 0 && !warnOnly {
				os.Exit(2) // the merge gate
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&basePath, "base", "", "base-branch fixtures TOML")
	cmd.Flags().StringVar(&headPath, "head", "", "PR fixtures TOML")
	cmd.Flags().Float64Var(&threshold, "threshold", 0.5, "similarity threshold")
	cmd.Flags().BoolVar(&warnOnly, "warn-only", false, "report failures without failing the check")
	cmd.Flags().IntVar(&concurrency, "concurrency", 6, "in-flight LLM calls")
	cmd.MarkFlagRequired("base")
	cmd.MarkFlagRequired("head")
	return cmd
}
