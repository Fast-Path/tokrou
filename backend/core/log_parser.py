"""Log file parsing utilities for Jarvis usage logs."""

import json
import re
from typing import Dict, Any, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


@dataclass
class UsageLogEntry:
    """Single usage log entry from Jarvis."""
    user_email: str
    timestamp: datetime
    session_id: str
    query_type: str
    complexity: str
    routed_to: str
    correct_route: bool
    input_tokens: int
    output_tokens: int
    prompt_tokens: int
    user_input_tokens: int
    model_used: str
    cost_usd: float
    delegated: bool
    delegation_model: str | None
    delegation_chain: List[str]
    tools_used: List[str]
    tool_count: int
    latency_ms: float
    time_to_first_token_ms: float
    conversation_depth: int
    file_types: List[str]
    file_sizes: List[int]
    file_count: int
    has_visual_content: bool
    retry_count: int
    cache_hit: bool
    error_occurred: bool
    error_type: str | None
    context_window_used: float


def parse_concatenated_json(raw_content: str) -> List[Dict[str, Any]]:
    """
    Parse concatenated JSON objects (no separators between them).
    
    Handles formats like:
    - {...}{...}{...}  (concatenated)
    - {...}\n{...}\n{...}  (JSONL)
    - [{...}, {...}]  (JSON array)
    
    Args:
        raw_content: Raw string containing JSON data
        
    Returns:
        List of parsed JSON objects
    """
    content = raw_content.strip()
    
    if not content:
        return []
    
    # Try JSON array first
    if content.startswith('['):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
    
    # Try JSONL (newline-delimited)
    if '\n' in content:
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        try:
            return [json.loads(line) for line in lines]
        except json.JSONDecodeError:
            pass
    
    # Handle concatenated JSON objects: {...}{...}
    # Split on }{ and reconstruct
    parts = re.split(r'\}\s*\{', content)
    
    if len(parts) == 1:
        # Single object
        return [json.loads(content)]
    
    # Reconstruct each object
    results = []
    for i, part in enumerate(parts):
        if i == 0:
            # First part: add closing brace
            obj_str = part + '}'
        elif i == len(parts) - 1:
            # Last part: add opening brace
            obj_str = '{' + part
        else:
            # Middle parts: add both braces
            obj_str = '{' + part + '}'
        
        results.append(json.loads(obj_str))
    
    return results


def load_usage_logs(filepath: Path) -> List[UsageLogEntry]:
    """
    Load usage logs from a file.
    
    Args:
        filepath: Path to the log file
        
    Returns:
        List of UsageLogEntry objects
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Log file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_content = f.read()
    
    entries = parse_concatenated_json(raw_content)
    return [dict_to_log_entry(entry) for entry in entries]


def dict_to_log_entry(data: Dict[str, Any]) -> UsageLogEntry:
    """Convert a dictionary to a UsageLogEntry."""
    return UsageLogEntry(
        user_email=data['user_email'],
        timestamp=datetime.fromisoformat(data['timestamp']),
        session_id=data['session_id'],
        query_type=data['query_type'],
        complexity=data['complexity'],
        routed_to=data['routed_to'],
        correct_route=data['correct_route'],
        input_tokens=data['input_tokens'],
        output_tokens=data['output_tokens'],
        prompt_tokens=data['prompt_tokens'],
        user_input_tokens=data['user_input_tokens'],
        model_used=data['model_used'],
        cost_usd=data['cost_usd'],
        delegated=data['delegated'],
        delegation_model=data.get('delegation_model'),
        delegation_chain=data.get('delegation_chain', []),
        tools_used=data.get('tools_used', []),
        tool_count=data['tool_count'],
        latency_ms=data['latency_ms'],
        time_to_first_token_ms=data['time_to_first_token_ms'],
        conversation_depth=data['conversation_depth'],
        file_types=data.get('file_types', []),
        file_sizes=data.get('file_sizes', []),
        file_count=data['file_count'],
        has_visual_content=data['has_visual_content'],
        retry_count=data['retry_count'],
        cache_hit=data['cache_hit'],
        error_occurred=data['error_occurred'],
        error_type=data.get('error_type'),
        context_window_used=data['context_window_used'],
    )
