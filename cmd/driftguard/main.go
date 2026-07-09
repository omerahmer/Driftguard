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

	cmd.AddCommand(list, create, version, promptDiffCmd(), promptHistoryCmd())
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
		Short: "Hand-label behavior diffs (gold ground truth); audits the judge against your labels",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			pool, err := connect(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()

			samples, err := core.SampleUnlabeledDiffs(ctx, pool, sampleSize)
			if err != nil {
				return err
			}
			reader := bufio.NewReader(cmd.InOrStdin())
			for i, s := range samples {
				// You judge independently; the judge's verdict is shown last and
				// only for reference, so you don't anchor to it.
				fmt.Printf("\n[%d/%d] input: %s\nexpected: %s\n--- before ---\n%s\n--- after ---\n%s\n",
					i+1, len(samples), s.Input, s.ExpectedBehavior, s.BeforeOutput, s.AfterOutput)
				fmt.Print("did behavior change? [y/n/s(kip)] ")
				line, _ := reader.ReadString('\n')
				switch strings.TrimSpace(strings.ToLower(line)) {
				case "y":
					if err := core.SetHumanLabel(ctx, pool, s.BehaviorDiffID, true); err != nil {
						return err
					}
				case "n":
					if err := core.SetHumanLabel(ctx, pool, s.BehaviorDiffID, false); err != nil {
						return err
					}
				}
			}

			audit, err := core.ComputeJudgeAudit(ctx, pool)
			if err != nil {
				return err
			}
			if jsonOut {
				return emit(audit)
			}
			if audit.Labeled == 0 {
				fmt.Println("\nno hand-labeled diffs yet")
				return nil
			}
			fmt.Printf("\njudge vs %d hand label(s): agreement %.0f%% | precision %.3f | recall %.3f\n",
				audit.Labeled, audit.Agreement*100, audit.Precision, audit.Recall)
			fmt.Printf("  confusion (human=truth): tp=%d fp=%d fn=%d tn=%d\n",
				audit.TP, audit.FP, audit.FN, audit.TN)
			return nil
		},
	}
	cmd.Flags().Int64Var(&sampleSize, "sample-size", 10, "diffs to sample (<=0 for all)")
	return cmd
}

func scoreCmd() *cobra.Command {
	var promptName, groundTruth string
	cmd := &cobra.Command{
		Use:   "score",
		Short: "Precision/recall/reduction across the threshold sweep",
		RunE: func(cmd *cobra.Command, _ []string) error {
			gt := core.GroundTruth(groundTruth)
			if gt != core.GroundTruthJudge && gt != core.GroundTruthHuman {
				return fmt.Errorf("--ground-truth must be %q or %q", core.GroundTruthJudge, core.GroundTruthHuman)
			}
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
			report, err := core.Score(ctx, pool, filter, core.DefaultThresholds(), gt)
			if err != nil {
				return err
			}
			if jsonOut {
				return emit(report)
			}
			fmt.Printf("ground truth: %s | samples: %d\n", report.GroundTruth, report.Samples)
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
	cmd.Flags().StringVar(&groundTruth, "ground-truth", "judge", "label source: judge | human")
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

func promptDiffCmd() *cobra.Command {
	var promptName, from, to string
	cmd := &cobra.Command{
		Use:   "diff",
		Short: "Diff two versions of a prompt (structure-aware)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			pool, err := connect(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()
			p, err := core.GetPromptByName(ctx, pool, promptName)
			if err != nil {
				return err
			}
			if p == nil {
				return core.PromptNotFoundError{Name: promptName}
			}
			vFrom, err := core.ResolveVersionRef(ctx, pool, p, from)
			if err != nil {
				return err
			}
			vTo, err := core.ResolveVersionRef(ctx, pool, p, to)
			if err != nil {
				return err
			}
			diff := core.ComputeDiff(vFrom.Content, vTo.Content)
			if jsonOut {
				return emit(diff)
			}
			for _, op := range diff.Ops {
				switch op.Type {
				case core.OpAdded:
					fmt.Printf("+ %s\n", op.Text)
				case core.OpRemoved:
					fmt.Printf("- %s\n", op.Text)
				default:
					fmt.Printf("  %s\n", op.Text)
				}
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&promptName, "prompt", "", "prompt name")
	cmd.Flags().StringVar(&from, "from", "", "source version ref")
	cmd.Flags().StringVar(&to, "to", "", "target version ref")
	cmd.MarkFlagRequired("prompt")
	cmd.MarkFlagRequired("from")
	cmd.MarkFlagRequired("to")
	return cmd
}

func promptHistoryCmd() *cobra.Command {
	var promptName string
	cmd := &cobra.Command{
		Use:   "history",
		Short: "List a prompt's versions, oldest first",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			pool, err := connect(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()
			p, err := core.GetPromptByName(ctx, pool, promptName)
			if err != nil {
				return err
			}
			if p == nil {
				return core.PromptNotFoundError{Name: promptName}
			}
			versions, err := core.ListVersions(ctx, pool, p.ID)
			if err != nil {
				return err
			}
			if jsonOut {
				return emit(versions)
			}
			for i, v := range versions {
				current := " "
				if p.CurrentVersionID != nil && v.ID == *p.CurrentVersionID {
					current = "*"
				}
				fmt.Printf("%s v%-3d %s  %s\n", current, i+1, v.ID.String()[:8], v.CreatedAt.Format("2006-01-02 15:04"))
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&promptName, "prompt", "", "prompt name")
	cmd.MarkFlagRequired("prompt")
	return cmd
}
