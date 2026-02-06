# Evaluation

NeMo Safe Synthesizer includes a comprehensive evaluation system that measures both the quality and privacy of generated synthetic data.

## Evaluation Components

| Component | Category | Description |
|-----------|----------|-------------|
| **Data Privacy Score** | Privacy | Overall privacy assessment |
| **PII Replay Detection** | Privacy | Checks if original PII values appear in synthetic data |
| **Membership Inference Protection** | Privacy | Measures resistance to membership inference attacks |
| **Attribute Inference Protection** | Privacy | Measures resistance to attribute inference attacks |
| **Column Distributions** | Quality | Statistical similarity of column distributions |
| **Correlations** | Quality | Preservation of column relationships |
| **Text Semantic Similarity** | Quality | Embedding-based semantic similarity |
| **Text Structure Similarity** | Quality | Structural similarity of text fields |
| **SQS Score** | Quality | Synthetic Quality Score (composite metric) |

## Evaluation Report

After evaluation, an HTML report is generated with interactive visualizations covering all enabled components.

The report is saved to the run's `generate/` directory as `evaluation_report.html`.

<!-- TODO: Add screenshot/example of evaluation report -->

## Configuring Evaluation

Evaluation components can be toggled via the `EvaluationParameters` config section.

<!-- TODO: Document which components can be enabled/disabled and their configuration options -->

## API Reference

- [:material-api: `Evaluator`](../reference/nemo_safe_synthesizer/evaluation/evaluator.md)
