"""BIORAG Eval v1 — Rule metrics + Qwen LLM-as-judge + manual calibration.

Architecture (3 layers):
  Layer 1: Rule metrics — deterministic, no LLM calls
  Layer 2: Qwen judge — LLM-as-judge for answer quality
  Layer 3: Manual calibration — review cards for human audit

No RAGAS dependency. No GPT API. Uses project's existing Qwen API (qwen-plus).
"""
