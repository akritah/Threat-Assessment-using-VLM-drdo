import json
import csv
from pathlib import Path
from typing import Any, Dict, List
import re

# Threat keywords definition for heuristics
HIGH_RISK_KEYWORDS = [
    "fire", "arson", "explosion", "shooting", "gun", "weapon", "knife",
    "assault", "abuse", "fighting", "riot", "burglary", "stealing", 
    "vandalism", "shoplifting", "theft", "break-in",
    # Added robust stems and firearm variants
    "firing", "fired", "shoot", "rifle", "rifles", "pistol", "pistols",
    "gunfire", "firearm", "firearms", "clash", "clashes", "violence", "violent",
    "machete", "hostage", "kidnap", "attacker", "intruder", "armed"
]

MEDIUM_RISK_KEYWORDS = [
    "suspicious", "running", "climbing", "arguing", "gathering", "chasing", 
    "hiding", "scuffling", "confrontation", "run", "climb", "chase", "hide"
]

CERTAINTY_KEYWORDS = [
    "clearly", "certain", "confirmed", "obvious", "unambiguous", "identifiable", 
    "high confidence", "evident"
]

UNCERTAINTY_KEYWORDS = [
    "possible", "perhaps", "maybe", "suggests", "unclear", "ambiguous", 
    "difficult to tell", "low confidence", "likely"
]

def parse_threat_level(threat_report: str) -> str:
    """
    Robustly parses the threat level from the VLM reasoning text.
    Uses regex matching and falls back to high-risk keywords for safety.
    """
    text_lower = threat_report.lower()
    
    # Direct regex patterns for explicit designations
    high_patterns = [
        r"threat level\s*[:\-=]?\s*high",
        r"threat\s*is\s*high",
        r"high\s*threat",
        r"danger level\s*[:\-=]?\s*high",
    ]
    medium_patterns = [
        r"threat level\s*[:\-=]?\s*medium",
        r"threat\s*is\s*medium",
        r"medium\s*threat",
        r"danger level\s*[:\-=]?\s*medium",
    ]
    
    if any(re.search(pat, text_lower) for pat in high_patterns):
        return "High"
    if any(re.search(pat, text_lower) for pat in medium_patterns):
        return "Medium"
        
    # Safety heuristic check: If there are weapons firing or active violence,
    # override to High/Medium threat level even if the model format varied.
    high_override_keywords = [
        "firing", "gunfire", "rifles", "weapons", "shooting", "assaulting", 
        "hostage", "armed confrontation", "clashes", "active gunfire"
    ]
    if any(kw in text_lower for kw in high_override_keywords):
        return "High"
        
    return "Low"

def calculate_threat_metrics(threat_level: str, frame_results: List[Dict[str, Any]], threat_report: str) -> Dict[str, Any]:
    """
    Computes threat score, evidence strength, and model confidence based on 
    deterministic textual heuristics and frame consistency.
    """
    # 1. Evidence Strength (fraction of frames with suspicious/threat keywords)
    anomalous_frames = 0
    total_frames = len(frame_results)
    
    for frame in frame_results:
        # Check description and actions for indicator keywords
        desc = str(frame.get("description", "")).lower()
        actions = str(frame.get("actions_occurring", "")).lower()
        combined = f"{desc} {actions}"
        
        has_anomaly = any(kw in combined for kw in HIGH_RISK_KEYWORDS + MEDIUM_RISK_KEYWORDS)
        if has_anomaly:
            anomalous_frames += 1
            
    evidence_strength = anomalous_frames / total_frames if total_frames > 0 else 0.0
    
    # 2. Model Confidence
    report_lower = threat_report.lower()
    certainty_hits = sum(1 for kw in CERTAINTY_KEYWORDS if kw in report_lower)
    uncertainty_hits = sum(1 for kw in UNCERTAINTY_KEYWORDS if kw in report_lower)
    
    # Determine confidence level
    if certainty_hits > uncertainty_hits:
        model_confidence = "High"
        confidence_val = 0.85 + (certainty_hits * 0.02)
    elif uncertainty_hits > certainty_hits:
        model_confidence = "Low"
        confidence_val = 0.40 - (uncertainty_hits * 0.02)
    else:
        model_confidence = "Medium"
        confidence_val = 0.65
        
    confidence_val = max(0.10, min(1.00, confidence_val))
    
    # 3. Threat Score calculation (0 to 100)
    level = threat_level.upper().strip()
    if "HIGH" in level:
        base_score = 75
    elif "MEDIUM" in level:
        base_score = 45
    else:
        base_score = 15
        
    # Scale base score by evidence strength and model confidence
    score_mod = (evidence_strength * 15) + (confidence_val * 10)
    
    # Keyword threat multipliers
    keyword_boost = 0
    if any(kw in report_lower for kw in ["weapon", "gun", "knife", "shooting"]):
        keyword_boost += 10
    if any(kw in report_lower for kw in ["fire", "arson", "explosion"]):
        keyword_boost += 10
        
    threat_score = int(base_score + score_mod + keyword_boost)
    threat_score = max(0, min(100, threat_score))
    
    # If it is Low threat and no evidence strength, cap at 30
    if "LOW" in level and evidence_strength == 0:
        threat_score = min(30, threat_score)
        
    return {
        "threat_score": threat_score,
        "evidence_strength": round(evidence_strength, 2),
        "model_confidence": model_confidence,
        "confidence_value": round(confidence_val, 2)
    }

def generate_timestamps(total_frames: int, duration_sec: float = None) -> List[str]:
    """Generates MM:SS format timestamps spaced evenly."""
    if duration_sec is None or duration_sec <= 0:
        # Fallback: assume 3 seconds between frames
        return [f"{i*3//60:02d}:{i*3%60:02d}" for i in range(total_frames)]
        
    timestamps = []
    for i in range(total_frames):
        curr_sec = int(i * duration_sec / (total_frames - 1)) if total_frames > 1 else 0
        timestamps.append(f"{curr_sec//60:02d}:{curr_sec%60:02d}")
    return timestamps

def build_event_timeline(frame_results: List[Dict[str, Any]], timestamps: List[str]) -> List[Dict[str, Any]]:
    """Builds a structured timeline list containing timestamped events."""
    timeline = []
    for idx, frame in enumerate(frame_results):
        desc = frame.get("description") or frame.get("environment_description") or "Scene observed"
        # Extract dynamic activity
        actions = frame.get("actions_occurring", [])
        if isinstance(actions, list) and actions:
            action_desc = ", ".join(actions)
        else:
            action_desc = str(actions)
            
        event_str = desc
        if action_desc and action_desc.lower() != "none" and action_desc not in desc:
            event_str = f"{desc} (Action: {action_desc})"
            
        # Extract threat score heuristic for this frame
        frame_text = f"{event_str} {str(frame.get('objects_present', ''))}".lower()
        frame_threat = 10
        if any(kw in frame_text for kw in HIGH_RISK_KEYWORDS):
            frame_threat = 85
        elif any(kw in frame_text for kw in MEDIUM_RISK_KEYWORDS):
            frame_threat = 50
            
        timeline.append({
            "timestamp": timestamps[idx] if idx < len(timestamps) else "00:00",
            "event": event_str,
            "threat_score": frame_threat
        })
    return timeline

def export_timeline(timeline: List[Dict[str, Any]], output_dir: Path, prefix: str = "timeline") -> Dict[str, Path]:
    """Saves timeline as JSON, CSV, and Markdown formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / f"{prefix}.json",
        "csv": output_dir / f"{prefix}.csv",
        "md": output_dir / f"{prefix}.md"
    }
    
    # 1. JSON
    with paths["json"].open("w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2, ensure_ascii=False)
        
    # 2. CSV
    with paths["csv"].open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Event Description", "Threat Score (0-100)"])
        for row in timeline:
            writer.writerow([row["timestamp"], row["event"], row["threat_score"]])
            
    # 3. Markdown
    with paths["md"].open("w", encoding="utf-8") as f:
        f.write("### 📅 Chronological Event Timeline\n\n")
        f.write("| Timestamp | Event Description | Threat Intensity |\n")
        f.write("| :---: | :--- | :---: |\n")
        for row in timeline:
            intensity = "🔴 High" if row["threat_score"] >= 75 else ("🟡 Medium" if row["threat_score"] >= 45 else "🟢 Low")
            f.write(f"| **{row['timestamp']}** | {row['event']} | {intensity} |\n")
            
    return paths

def generate_explainable_report(
    video_name: str,
    action_caption: str,
    threat_report: str,
    threat_level: str,
    metrics: Dict[str, Any],
    timeline: List[Dict[str, Any]],
    frame_results: List[Dict[str, Any]],
    output_dir: Path,
    prefix: str = "report"
) -> Dict[str, Path]:
    """Generates complete explainable report in JSON and Markdown formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / f"{prefix}.json",
        "md": output_dir / f"{prefix}.md"
    }
    
    # Recommendations mappings
    dispatch = "No emergency dispatch required."
    recs = ["Maintain routine surveillance scans.", "Log incident context in database."]
    
    act_lower = action_caption.lower()
    if any(k in act_lower for k in ["fire", "arson", "explosion"]):
        dispatch = "🚒 Fire Rescue & Emergency Medical Teams Dispatched"
        recs = ["Evacuate nearby personnel immediately.", "Alert local fire station departments.", "Activate localized exhaust ventilation."]
    elif any(k in act_lower for k in ["shooting", "gun", "weapon"]):
        dispatch = "🚨 Armed Police & SWAT Tactical Containment Dispatched"
        recs = ["Initiate building lockdown procedures.", "Advise security personnel to seek safe cover.", "Establish visual feed telemetry logs for dispatch squad."]
    elif any(k in act_lower for k in ["assault", "abuse", "fighting", "riot"]):
        dispatch = "🚓 Police Patrol & Local Security Dispatched"
        recs = ["Dispatch immediate ground guards to contain situation.", "Alert police patrol units.", "Keep continuous zoom tracking on primary suspects."]
    elif any(k in act_lower for k in ["burglary", "stealing", "vandalism", "shoplifting"]):
        dispatch = "🚓 Local Police Dispatched to secure the property"
        recs = ["Lock all digital exits to restrict runaway routes.", "Secure entry/exit logs.", "Prepare evidence frames packaging export for police."]

    # 1. JSON Export
    report_dict = {
        "metadata": {
            "video_source": video_name,
            "total_frames_analyzed": len(frame_results),
            "inference_backend": "Two-Stage Hybrid VLM"
        },
        "assessment": {
            "threat_level": threat_level,
            "threat_score": metrics["threat_score"],
            "evidence_strength": metrics["evidence_strength"],
            "model_confidence": metrics["model_confidence"]
        },
        "response": {
            "emergency_dispatch": dispatch,
            "operational_recommendations": recs
        },
        "timeline": timeline,
        "observations": {
            "scene_summary": action_caption,
            "detailed_reasoning": threat_report
        },
        "evidence_frames": [frame.get("image", "") for frame in frame_results]
    }
    with paths["json"].open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
    # 2. Markdown Export
    with paths["md"].open("w", encoding="utf-8") as f:
        f.write(f"# 🛡️ DRDO Surveillance Threat Assessment Report\n\n")
        f.write(f"## 1. Scene Summary\n")
        f.write(f"*   **Video Source File:** `{video_name}`\n")
        f.write(f"*   **Observed Activity Caption:** {action_caption}\n\n")
        
        f.write(f"## 2. Threat Metrics Dashboard\n")
        f.write(f"| Metric | Assessment Value | Interpretation |\n")
        f.write(f"| :--- | :---: | :--- |\n")
        f.write(f"| **Estimated Threat Level** | `{threat_level}` | Operational Response priority |\n")
        f.write(f"| **Continuous Threat Score** | **{metrics['threat_score']}/100** | Scaled danger classification |\n")
        f.write(f"| **Evidence Frame Strength** | **{int(metrics['evidence_strength']*100)}%** | Ratio of anomalous frame matches |\n")
        f.write(f"| **VLM Model Confidence** | `{metrics['model_confidence']}` | Certainty of textual indicators |\n\n")
        
        f.write(f"## 3. Chronological Event Timeline\n")
        f.write(f"| Timestamp | Event Description | Threat Intensity |\n")
        f.write(f"| :---: | :--- | :---: |\n")
        for row in timeline:
            intensity = "🔴 High" if row["threat_score"] >= 75 else ("🟡 Medium" if row["threat_score"] >= 45 else "🟢 Low")
            f.write(f"| **{row['timestamp']}** | {row['event']} | {intensity} |\n")
        f.write("\n")
        
        f.write(f"## 4. Explainable System Reasoning\n")
        f.write(f"{threat_report}\n\n")
        
        f.write(f"## 5. Operational Response & Dispatch\n")
        f.write(f"🚨 **Emergency Action:** `{dispatch}`\n\n")
        f.write(f"### Recommended Actions:\n")
        for rec in recs:
            f.write(f"*   {rec}\n")
        f.write("\n")
        
        f.write(f"## 6. Physical Evidence Logs\n")
        for i, frame in enumerate(frame_results, 1):
            path_str = frame.get("image", "")
            f.write(f"*   **Frame {i}:** [{Path(path_str).name}](file:///{path_str.replace('\\\\', '/').replace('\\', '/')})\n")
            
    return paths

def infer_threat_level(text: str) -> str:
    """Robustly infers the threat level (Low/Medium/High) by calling parse_threat_level."""
    return parse_threat_level(text)

