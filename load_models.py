from transformers import pipeline
from rouge_score import rouge_scorer

bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
t5_summarizer = pipeline("summarization", model="t5-small")

def get_real_summaries(text):
    bart_summary = bart_summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    t5_summary = t5_summarizer("summarize: " + text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_bart = scorer.score(text, bart_summary)['rougeL'].fmeasure
    rouge_t5 = scorer.score(text, t5_summary)['rougeL'].fmeasure

    return {
        "gpt": "GPT-3.5 not available in this version. Support coming soon.",
        "bart": bart_summary,
        "t5": t5_summary,
        "bart_metrics": {
            "rougeL": round(rouge_bart, 3),
            "length": len(bart_summary.split())
        },
        "t5_metrics": {
            "rougeL": round(rouge_t5, 3),
            "length": len(t5_summary.split())
        }
    }
