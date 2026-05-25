#!/usr/bin/env python3
"""
Hermès Memory Provider — 本地 JSON + Vector 混合儲存
Inspired by agentmemory architecture (rohitg00/agentmemory)

Pipeline: Session → RawObservation → CompressedObservation → Memory/Lesson
"""

import os
import json
import time
import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ── 基礎路徑設定 ──────────────────────────────────────
SKILL_DIR = Path(__file__).parent
HERMES_DIR = Path.home() / ".hermes"
MEMORY_DIR = HERMES_DIR / "memory"
OBS_DIR = MEMORY_DIR / "observations"      # RawObservation 暫存
MEM_DIR = MEMORY_DIR / "memories"          # 持久化 Memory
LESSON_DIR = MEMORY_DIR / "lessons"        # Lesson 結晶
SESSION_DIR = MEMORY_DIR / "sessions"       # Session 元數據
CONFIG_DIR = HERMES_DIR / "config"

for d in [OBS_DIR, MEM_DIR, LESSON_DIR, SESSION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 工具函式 ────────────────────────────────────────────
def now_iso():
    return datetime.utcnow().isoformat()

def uuid4():
    return str(uuid.uuid4())

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ── Session 管理 ────────────────────────────────────────
def session_start(session_id: str, cwd: str, project: str = "default", model: str = "") -> dict:
    """新建 Session 或返回現有活躍 Session"""
    session_file = SESSION_DIR / f"{session_id}.json"
    
    session = {
        "id": session_id,
        "project": project,
        "cwd": cwd,
        "startedAt": now_iso(),
        "status": "active",
        "observationCount": 0,
        "model": model,
        "tags": [],
    }
    
    save_json(session_file, session)
    return session

def session_end(session_id: str) -> dict:
    """標記 Session 結束，觸發可選的自動 Consolidate"""
    session_file = SESSION_DIR / f"{session_id}.json"
    session = load_json(session_file)
    
    session["endedAt"] = now_iso()
    session["status"] = "completed"
    save_json(session_file, session)
    
    return session

# ── RawObservation 捕獲 ─────────────────────────────────
def observe(
    session_id: str,
    hook_type: str,
    data: dict,
    cwd: str = ".",
    project: str = "default",
) -> dict:
    """
    捕獲 RawObservation（來自 hook 觸發）
    
    hook_type: prompt_submit | post_tool_use | post_tool_failure | task_completed | session_end
    data: 任務相關數據（prompt、tool_input、tool_output 等）
    """
    obs_id = uuid4()
    obs = {
        "id": obs_id,
        "sessionId": session_id,
        "timestamp": now_iso(),
        "hookType": hook_type,
        "cwd": cwd,
        "project": project,
        "data": data,
        "compressed": False,
    }
    
    obs_file = OBS_DIR / f"{obs_id}.json"
    save_json(obs_file, obs)
    
    # 更新 session observationCount
    session_file = SESSION_DIR / f"{session_id}.json"
    if session_file.exists():
        session = load_json(session_file)
        session["observationCount"] = session.get("observationCount", 0) + 1
        save_json(session_file, session)
    
    return obs

# ── CompressedObservation（語義壓縮）─────────────────────
def compress_observation(obs_id: str, summary: str, facts: list, concepts: list, importance: float = 0.5) -> dict:
    """
    將 RawObservation 壓縮為 CompressedObservation
    這裡是簡化版本 — 真實實現需要 LLM 呼叫
    """
    obs_file = OBS_DIR / f"{obs_id}.json"
    obs = load_json(obs_file)
    
    compressed = {
        "id": obs_id,
        "sessionId": obs["sessionId"],
        "timestamp": obs["timestamp"],
        "type": obs["hookType"],
        "title": summary[:100] if summary else "Untitled",
        "facts": facts,
        "concepts": concepts,
        "importance": importance,
        "rawData": obs["data"],
        "compressedAt": now_iso(),
    }
    
    obs.update({"compressed": True, "compressedData": compressed})
    save_json(obs_file, obs)
    
    return compressed

# ── Memory 寫入 ─────────────────────────────────────────
MEMORY_TYPES = ["pattern", "preference", "architecture", "bug", "workflow", "fact"]

def create_memory(
    memory_type: str,
    title: str,
    content: str,
    facts: list = None,
    concepts: list = None,
    files: list = None,
    session_ids: list = None,
    strength: float = 1.0,
    entities: list = None,   # SAGE-ready: (subject, relation, object) triplets
) -> dict:
    """
    創建持久化 Memory
    
    SAGE-Ready: entities 欄位支持結構化三元組
    {
      "subject": "Pokemon TCG",
      "relation": "has_attribute",
      "object": "PSA 10 population",
      "confidence": 0.9
    }
    當 SAGE 官方代碼開源後，可無縫升級至 Graph Memory
    """
    if memory_type not in MEMORY_TYPES:
        raise ValueError(f"Invalid type. Must be one of {MEMORY_TYPES}")
    
    mem_id = uuid4()
    memory = {
        "id": mem_id,
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
        "type": memory_type,
        "title": title,
        "content": content,
        "facts": facts or [],
        "concepts": concepts or [],
        "entities": entities or [],   # SAGE-ready entity triplets
        "files": files or [],
        "sessionIds": session_ids or [],
        "strength": strength,
        "version": 1,
        "isLatest": True,
        "accessCount": 0,
        "lastAccessed": now_iso(),
        "downstreamHits": 0,   # SAGE feedback: times this memory was actually used in response
    }
    
    mem_file = MEM_DIR / f"{mem_id}.json"
    save_json(mem_file, memory)
    
    # 更新 index
    _update_memory_index(memory)
    
    return memory

def _update_memory_index(memory: dict):
    """更新 Memory index（用於快速檢索）"""
    index_file = MEM_DIR / "_index.json"
    index = load_json(index_file)
    
    index[memory["id"]] = {
        "type": memory["type"],
        "title": memory["title"],
        "concepts": memory["concepts"],
        "strength": memory["strength"],
    }
    
    save_json(index_file, index)

# ── Lesson 提煉 ─────────────────────────────────────────
def create_lesson(
    content: str,
    context: str,
    source: str = "manual",  # "crystal" | "manual" | "consolidation"
    tags: list = None,
    project: str = "default",
) -> dict:
    """從 Memory 提煉出 Lesson（教訓）"""
    lesson_id = uuid4()
    lesson = {
        "id": lesson_id,
        "content": content,
        "context": context,
        "confidence": 0.7,
        "reinforcements": 0,
        "source": source,
        "project": project,
        "tags": tags or [],
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
        "lastReinforcedAt": None,
        "decayRate": 0.05,
    }
    
    lesson_file = LESSON_DIR / f"{lesson_id}.json"
    save_json(lesson_file, lesson)
    
    return lesson

# ── 簡單關鍵詞檢索（BM25 簡化版）─────────────────────────
def search_memories(
    query: str,
    top_k: int = 5,
    memory_type: str = None,
    time_range: tuple = None,    # SAGE: (start_iso, end_iso) hard constraint
    concepts_filter: list = None,  # SAGE: hard constraint filter
    use_downstream_feedback: bool = False,  # SAGE: boost memories that were actually used
) -> list:
    """
    簡化版關鍵詞檢索（支援 Hybrid Search 過渡）
    
    SAGE-Ready 升級：
    1. time_range (start, end): 硬性時間邊界過濾
    2. concepts_filter: 硬性範疇過濾
    3. downstreamHits 加權：被下游採用的記憶 boosted
    
    完整版應使用：BM25 + Vector Embedding + Knowledge Graph
    """
    index_file = MEM_DIR / "_index.json"
    index = load_json(index_file)
    
    results = []
    query_lower = query.lower()
    query_words = query_lower.split()
    
    for mem_id, meta in index.items():
        mem_file = MEM_DIR / f"{mem_id}.json"
        if not mem_file.exists():
            continue
        
        memory = load_json(mem_file)
        
        # ── SAGE 硬性約束過濾（時間邊界）──────────────────
        if time_range:
            start_iso, end_iso = time_range
            created = memory.get("createdAt", "")
            if created and created < start_iso:
                continue
            if created and created > end_iso:
                continue
        
        # ── SAGE 硬性約束過濾（範疇）─────────────────────
        if concepts_filter:
            mem_concepts = set(c.lower() for c in memory.get("concepts", []))
            if not any(c.lower() in mem_concepts for c in concepts_filter):
                continue
        
        # ── BM25 關鍵詞匹配 ──────────────────────────────
        content_lower = (memory.get("content", "") + " " + memory.get("title", "")).lower()
        score = sum(1 for w in query_words if w in content_lower)
        
        # 額外檢查 concepts
        for concept in memory.get("concepts", []):
            if any(w in concept.lower() for w in query_words):
                score += 2
        
        # ── SAGE 下游反饋加權 ──────────────────────────
        if use_downstream_feedback:
            downstream_hits = memory.get("downstreamHits", 0)
            score *= (1 + downstream_hits * 0.1)  # 每被採用一次 +10% boost
        
        # 強度加成
        score *= (1 + memory.get("strength", 0.5) * 0.5)
        
        if score > 0 and (memory_type is None or memory.get("type") == memory_type):
            results.append({
                "memory": memory,
                "score": score,
            })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def record_downstream_hit(memory_id: str) -> dict:
    """
    SAGE 下游反饋：當記憶被實際採用進 response 時呼叫
    用於日後 decay 決策：長期未被引用的記憶加速衰退
    """
    mem_file = MEM_DIR / f"{memory_id}.json"
    if not mem_file.exists():
        return {"error": "memory not found"}
    
    memory = load_json(mem_file)
    memory["downstreamHits"] = memory.get("downstreamHits", 0) + 1
    memory["lastDownstreamHit"] = now_iso()
    save_json(mem_file, memory)
    
    return {"id": memory_id, "downstreamHits": memory["downstreamHits"]}

# ── Memory Decay（遺忘權重）──────────────────────────────
def apply_decay(days: int = 7, min_strength: float = 0.2):
    """
    對所有 Memory 套用 decay
    基於時間衰減強度，低於閾值則標記為可刪除
    """
    for mem_file in MEM_DIR.glob("*.json"):
        if mem_file.name.startswith("_"):
            continue
        
        memory = load_json(mem_file)
        created = datetime.fromisoformat(memory["createdAt"])
        age_days = (datetime.utcnow() - created).days
        
        if age_days < days:
            continue
        
        # 指數衰減
        decay_amount = memory.get("decayRate", 0.05) * age_days
        new_strength = max(memory.get("strength", 1.0) - decay_amount, 0.1)
        
        memory["strength"] = new_strength
        memory["updatedAt"] = now_iso()
        
        if new_strength < min_strength:
            memory["forgetAfter"] = now_iso()
        
        save_json(mem_file, memory)

# ── 健康檢查 ────────────────────────────────────────────
def health_check() -> dict:
    obs_count = len(list(OBS_DIR.glob("*.json")))
    mem_count = len([f for f in MEM_DIR.glob("*.json") if not f.name.startswith("_")])
    lesson_count = len(list(LESSON_DIR.glob("*.json")))
    session_count = len(list(SESSION_DIR.glob("*.json")))
    
    return {
        "status": "healthy",
        "timestamp": now_iso(),
        "observations": obs_count,
        "memories": mem_count,
        "lessons": lesson_count,
        "sessions": session_count,
        "storage_dir": str(MEMORY_DIR),
    }

# ── 導出給 Hermès 使用 ──────────────────────────────────
OBSERVE_HOOK_TYPES = [
    "prompt_submit",
    "post_tool_use",
    "post_tool_failure",
    "task_completed",
    "session_end",
    "pre_compact",
]

__all__ = [
    "session_start", "session_end",
    "observe",
    "compress_observation",
    "create_memory", "search_memories", "record_downstream_hit",
    "create_lesson",
    "apply_decay",
    "health_check",
    "OBSERVE_HOOK_TYPES",
]