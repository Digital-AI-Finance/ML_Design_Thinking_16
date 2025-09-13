# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

LaTeX/Beamer course materials for "Machine Learning for Smarter Innovation" - a BSc-level 10-week course (310 slides) demonstrating how ML/AI augments design thinking processes through innovation discovery rather than user analysis.

## Code Architecture

### Directory Structure
```
ML_Design_Thinking_16/
├── Week_01/                        # COMPLETE - Full implementation with Priority 1&2 improvements
│   ├── 20250913_1430_week01_priority2_partial.tex  # Latest version (52 pages)
│   ├── charts/                     # 20+ ML visualizations (all innovation-focused)
│   ├── scripts/                    # Python generators using sklearn/matplotlib
│   └── previous/                   # Version history
├── Week_02-10/                     # PLANNED - Structure ready, content pending
├── ML_Design_Course/               # Course-wide resources
│   ├── course_visuals/            # Shared visualizations
│   └── full_course_content_toc.md # Complete 310-slide specification
└── *.py                           # Automation tools
```

### Python Chart Generation Architecture
All visualizations use real ML algorithms with reproducible seeds:
- **Base pattern**: sklearn algorithms → matplotlib visualization → PDF/PNG output
- **Data scale**: 1000-10000 points per visualization
- **Standard size**: `figsize=(14, 10)` for consistency
- **Color scheme**: `mlblue=#1f77b4`, `mlorange=#ff7f0e`, `mlgreen=#2ca02c`, `mlred=#d62728`, `mlpurple=#9467bd`

### LaTeX Slide Structure
Each week follows a 49-slide structure with consistent formatting:
```latex
% Part structure (31 base slides + transitions + dividers = ~49 total)
% 1. Opening Power Chart (1 slide)
% 2. Part 1: Foundation (3 slides + divider)
% 3. Part 2: Technical ML (10 slides + divider)
% 4. Part 3: Innovation Pattern Analysis (8 slides + divider)
% 5. Part 4: Summary & Practice (5 slides + divider)
% 6. Appendix: Technical Details (4-6 slides)
```

## Essential Commands

```bash
# Build and compile slides
cd Week_01
python ../compile_slides.py 20250913_1430_week01_priority2_partial.tex

# Quality control
python ../check_font_sizes.py *.tex      # Verify exactly 2 font sizes
python ../fix_overfull_charts.py *.tex   # Auto-fix overfull boxes

# Chart generation
cd scripts
python create_convergence_flow.py        # Individual chart
python generate_all_charts.py            # Batch generation (if exists)

# Manual LaTeX compilation
pdflatex -interaction=nonstopmode filename.tex
pdflatex -jobname=filename_v2 filename.tex  # If PDF locked

# Cleanup auxiliary files
mkdir -p temp && move *.aux *.log *.nav *.out *.snm *.toc *.vrb temp/ 2>nul
```

## Critical Content Alignment

### Innovation-Focused Terminology (MANDATORY)
The course explores ML for **innovation discovery**, not user understanding:

| ❌ Never Use | ✅ Always Use |
|--------------|---------------|
| User segments | Innovation categories |
| User personas | Innovation archetypes |
| User needs/pain points | Innovation opportunities |
| User journey | Innovation evolution |
| Empathy maps | Pattern maps |
| User behavior | Innovation patterns |

### LaTeX Technical Requirements
- **Font sizes**: Exactly 2 (`\Large` for titles, `\normalsize` for content)
- **No small fonts**: Never use `\small`, `\tiny`, `\footnotesize`
- **Column layouts**: `0.43/0.55` split with charts always on right
- **Chart widths**: `0.85\textwidth` (full), `0.75` (medium), `0.65` (sidebar)
- **No Unicode**: ASCII only, use `` `` for quotes
- **Colors**: Use predefined `mlblue`, `mlorange`, `mlgreen`, `mlred`, `mlpurple`

## Week 1 Specific Implementation

### Current Status
- **52 pages** with complete Priority 1 fixes and 50% Priority 2 enhancements
- **20+ charts** all displaying correctly (no placeholders)
- **Innovation-aligned** throughout (no user-centric language)
- **Production ready** with comparison tables, decision trees, Python code examples

### Key Improvements Implemented
1. Algorithm comparison table (slide 17)
2. Decision tree for algorithm selection (slide 18)
3. Evaluation metrics comparison (slide 21)
4. Python code examples in appendix (slide 50)
5. All 6 missing innovation charts created

### Charts Created for Week 1
```
innovation_patterns_visual.pdf    # Market segments with opportunities
innovation_archetypes.pdf         # Five innovation types
opportunity_zones.pdf              # White space identification
innovation_adjacencies.pdf         # Innovation synergy network
opportunity_heatmap.pdf            # Scoring matrix
clustering_decision_tree.pdf      # Algorithm selection guide
evaluation_metrics_comparison.pdf  # Metrics behavior comparison
```

## Development Workflow

1. **Timestamp versions**: `YYYYMMDD_HHMM_description.tex`
2. **Generate charts first**: Create visualizations before slide content
3. **Quality checks**: Run font/overfull checks before compilation
4. **Version control**: Move old versions to `previous/`
5. **Clean builds**: Use `compile_slides.py` for consistent output

## Course Structure Reference

- **Week 1**: Clustering for Innovation Discovery (COMPLETE)
- **Week 2**: Advanced Clustering Techniques
- **Week 3**: NLP for Context Understanding
- **Week 4**: Classification for Problem Definition
- **Week 5**: Topic Modeling for Hidden Themes
- **Week 6**: Generative AI for Ideation
- **Week 7**: SHAP for Feature Prioritization
- **Week 8**: Structured Generation
- **Week 9**: Multi-Metric Validation
- **Week 10**: Testing, Ethics & Evolution

## Key Resources

- Course outline: `AI_Innovation_Course_Outline.md` (625 lines)
- Full TOC: `ML_Design_Course/full_course_content_toc.md` (588 lines)
- Week 1 status: `Week_01/FINAL_IMPROVEMENT_REPORT.md`