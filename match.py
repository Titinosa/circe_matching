#!/usr/bin/env python3
"""
match.py — Pure Python deterministic matching (zero LLM)

Reads profiles.json, computes pairwise match scores, allocates exactly 4
matches per attendee subject to constraints, validates, and saves matches.json.

Usage:
    python match.py --input profiles.json --output matches.json
"""

import argparse
import json
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------
POINTS_INVESTOR_FOUNDER_THESIS = 4
POINTS_SECTOR_OVERLAP = 3
POINTS_STAGE_ALIGNMENT = 3
POINTS_FUNDRAISING_INVESTOR = 3
POINTS_HIRING_OPERATOR = 2
POINTS_TECH_NEED_ENGINEERING = 2
POINTS_SHARED_TOP_OF_MIND = 2

# Trust accelerators (tie-breakers)
POINTS_SHARED_SCHOOL = 0.5
POINTS_SHARED_COMPANY = 1
POINTS_SHARED_PIVOT = 1
POINTS_SHARED_INTEREST = 1

# Allocation
MATCHES_PER_PERSON = 4
SOFT_CAP = 3  # prefer ≤3 recommendations
HARD_CAP = 4  # absolute max recommendations

# Top-of-mind values to exclude from matching
EXCLUDED_TOP_OF_MIND = {"Just want to meet cool women", "Community/relationships"}


def normalize_role(role_type):
    """Normalize role_type to one of: Founder, Operator, Investor, Exploring."""
    if not role_type:
        return "Exploring"
    role = str(role_type).strip()
    role_lower = role.lower()
    if "founder" in role_lower:
        return "Founder"
    if "operator" in role_lower:
        return "Operator"
    if "investor" in role_lower:
        return "Investor"
    if "explor" in role_lower or "transition" in role_lower:
        return "Exploring"
    return "Exploring"


def normalize_stage(stage):
    """Normalize stage strings for comparison."""
    if not stage:
        return None
    s = str(stage).strip().lower()
    # Map common variants
    mappings = {
        "pre-seed": "preseed",
        "preseed": "preseed",
        "pre seed": "preseed",
        "seed": "seed",
        "series a": "series_a",
        "series b": "series_b",
        "series c": "series_c",
        "series d": "series_d",
        "growth": "growth",
        "public": "public",
        "revenue": "revenue",
    }
    for key, val in mappings.items():
        if key in s:
            return val
    return s


def normalize_sector(sector):
    """Normalize sector string for comparison."""
    if not sector:
        return None
    return str(sector).strip().lower().replace("/", "").replace(" ", "")


def get_sectors_set(profile):
    """Get normalized set of sectors."""
    sectors = profile.get("sectors") or []
    return {normalize_sector(s) for s in sectors if s}


def get_stage(profile):
    """Get normalized stage."""
    return normalize_stage(profile.get("stage"))


def get_top_of_mind(profile):
    """Get meaningful top-of-mind set (excluding generic ones)."""
    items = profile.get("top_of_mind") or []
    return {
        str(item).strip()
        for item in items
        if item and str(item).strip() not in EXCLUDED_TOP_OF_MIND
    }


def get_functional_strengths(profile):
    """Get set of functional strengths."""
    strengths = profile.get("functional_strengths") or []
    return {str(s).strip().lower() for s in strengths if s}


def get_completed_schools(profile):
    """Get list of completed school dicts."""
    schools = profile.get("schools") or []
    return [
        s for s in schools
        if isinstance(s, dict) and s.get("name") and s.get("completed") is True
    ]


def get_all_schools(profile):
    """Get list of all school dicts."""
    schools = profile.get("schools") or []
    return [s for s in schools if isinstance(s, dict) and s.get("name")]


def get_notable_companies(profile):
    """Get set of notable companies (lowered)."""
    companies = profile.get("notable_companies") or []
    return {str(c).strip().lower() for c in companies if c}


def get_career_pivots(profile):
    """Get set of career pivots (lowered)."""
    pivots = profile.get("career_pivots") or []
    return {str(p).strip().lower() for p in pivots if p}


def get_explicit_interests(profile):
    """Get set of explicit interests (lowered)."""
    interests = profile.get("explicit_interests") or []
    return {str(i).strip().lower() for i in interests if i}


def get_investor_profile(profile):
    """Get investor profile dict or None."""
    ip = profile.get("investor_profile")
    if isinstance(ip, dict):
        return ip
    return None


def schools_overlap_hard_avoid(s1, s2):
    """Check if two school entries trigger the hard avoid rule.
    Hard avoid if ALL THREE: same school, same discipline, overlapping years.
    """
    if not s1.get("name") or not s2.get("name"):
        return False
    if s1["name"].strip().lower() != s2["name"].strip().lower():
        return False
    # Same discipline?
    d1 = s1.get("discipline") or s1.get("degree")
    d2 = s2.get("discipline") or s2.get("degree")
    if not d1 or not d2:
        return False  # Can't determine → don't avoid
    if str(d1).strip().lower() != str(d2).strip().lower():
        return False
    # Overlapping years?
    start1 = s1.get("start_year")
    end1 = s1.get("end_year")
    start2 = s2.get("start_year")
    end2 = s2.get("end_year")
    if start1 is None or end1 is None or start2 is None or end2 is None:
        return False  # Can't determine → don't avoid
    try:
        start1, end1, start2, end2 = int(start1), int(end1), int(start2), int(end2)
    except (ValueError, TypeError):
        return False
    # Overlapping if not (one ends before the other starts)
    return not (end1 < start2 or end2 < start1)


def check_hard_avoid(p1, p2):
    """Check if pair should be hard-avoided."""
    schools1 = get_all_schools(p1)
    schools2 = get_all_schools(p2)
    for s1 in schools1:
        for s2 in schools2:
            if schools_overlap_hard_avoid(s1, s2):
                return True
    return False


def shared_completed_school(p1, p2):
    """Check if two profiles share a completed past school. Returns bool."""
    schools1 = get_completed_schools(p1)
    schools2 = get_completed_schools(p2)
    names1 = {s["name"].strip().lower() for s in schools1}
    names2 = {s["name"].strip().lower() for s in schools2}
    return bool(names1 & names2)


def shared_current_school(p1, p2):
    """Check if they share a current (non-completed) school."""
    schools1 = get_all_schools(p1)
    schools2 = get_all_schools(p2)
    current1 = {
        s["name"].strip().lower()
        for s in schools1
        if s.get("completed") is not True
    }
    current2 = {
        s["name"].strip().lower()
        for s in schools2
        if s.get("completed") is not True
    }
    return bool(current1 & current2)


def compute_pairwise_score(p1, p2):
    """Compute the match score between two profiles. Returns (score, reasons)."""
    score = 0.0
    reasons = []
    role1 = normalize_role(p1.get("role_type"))
    role2 = normalize_role(p2.get("role_type"))
    sectors1 = get_sectors_set(p1)
    sectors2 = get_sectors_set(p2)
    stage1 = get_stage(p1)
    stage2 = get_stage(p2)
    tom1 = get_top_of_mind(p1)
    tom2 = get_top_of_mind(p2)
    strengths1 = get_functional_strengths(p1)
    strengths2 = get_functional_strengths(p2)

    # --- Core Alignment ---

    # Investor thesis ↔ founder need
    inv, fnd = None, None
    if role1 == "Investor" and role2 == "Founder":
        inv, fnd = p1, p2
    elif role2 == "Investor" and role1 == "Founder":
        inv, fnd = p2, p1

    if inv and fnd:
        ip = get_investor_profile(inv)
        if ip:
            inv_sectors = {normalize_sector(s) for s in (ip.get("sector_focus") or []) if s}
            inv_stages = {normalize_stage(s) for s in (ip.get("stage_focus") or []) if s}
            fnd_sectors = get_sectors_set(fnd)
            fnd_stage = get_stage(fnd)

            sector_overlap = inv_sectors & fnd_sectors
            stage_match = fnd_stage and fnd_stage in inv_stages

            if sector_overlap or stage_match:
                score += POINTS_INVESTOR_FOUNDER_THESIS
                overlap_details = []
                if sector_overlap:
                    overlap_details.append(f"sector: {', '.join(sector_overlap)}")
                if stage_match:
                    overlap_details.append(f"stage: {fnd_stage}")
                reasons.append(
                    f"Investor thesis \u2194 founder need ({', '.join(overlap_details)})"
                )

    # Sector overlap
    common_sectors = sectors1 & sectors2
    if common_sectors:
        score += POINTS_SECTOR_OVERLAP
        reasons.append(f"Sector overlap: {', '.join(common_sectors)}")

    # Stage alignment
    if stage1 and stage2 and stage1 == stage2:
        score += POINTS_STAGE_ALIGNMENT
        reasons.append(f"Stage alignment: {stage1}")

    # Fundraising founder ↔ aligned investor
    if inv and fnd:
        fnd_fundraising = fnd.get("is_fundraising") is True
        if fnd_fundraising:
            ip = get_investor_profile(inv)
            if ip:
                thesis = ip.get("thesis_clarity", "Low")
                if thesis != "Low":
                    score += POINTS_FUNDRAISING_INVESTOR
                    reasons.append("Fundraising founder \u2194 investor")

    # Hiring founder ↔ scaling operator
    for pa, pb in [(p1, p2), (p2, p1)]:
        ra = normalize_role(pa.get("role_type"))
        rb = normalize_role(pb.get("role_type"))
        if ra == "Founder" and pa.get("is_hiring") is True and rb == "Operator":
            if "operations" in get_functional_strengths(pb) or "growth" in get_functional_strengths(pb):
                score += POINTS_HIRING_OPERATOR
                reasons.append(f"Hiring founder \u2194 scaling operator")
                break

    # Technical need ↔ engineering background
    for pa, pb in [(p1, p2), (p2, p1)]:
        if pa.get("has_technical_needs") is True and "engineering" in get_functional_strengths(pb):
            score += POINTS_TECH_NEED_ENGINEERING
            reasons.append("Technical need \u2194 engineering background")
            break

    # Shared meaningful top-of-mind
    shared_tom = tom1 & tom2
    if shared_tom:
        score += POINTS_SHARED_TOP_OF_MIND
        reasons.append(f"Shared top-of-mind: {', '.join(shared_tom)}")

    # --- Trust Accelerators ---

    # Shared completed past school
    if shared_completed_school(p1, p2):
        score += POINTS_SHARED_SCHOOL
        reasons.append("Shared completed school")

    # Shared current school = 0 points (no points, but track)

    # Shared notable company
    companies1 = get_notable_companies(p1)
    companies2 = get_notable_companies(p2)
    shared_companies = companies1 & companies2
    if shared_companies:
        score += POINTS_SHARED_COMPANY
        reasons.append(f"Shared company: {', '.join(shared_companies)}")

    # Shared pivot pattern
    pivots1 = get_career_pivots(p1)
    pivots2 = get_career_pivots(p2)
    shared_pivots = pivots1 & pivots2
    if shared_pivots:
        score += POINTS_SHARED_PIVOT
        reasons.append(f"Shared pivot: {', '.join(shared_pivots)}")

    # Shared explicit interest
    interests1 = get_explicit_interests(p1)
    interests2 = get_explicit_interests(p2)
    shared_interests = interests1 & interests2
    if shared_interests:
        score += POINTS_SHARED_INTEREST
        reasons.append(f"Shared interest: {', '.join(list(shared_interests)[:3])}")

    return score, reasons


def classify_match_type(p1, p2, score, reasons):
    """Determine the match type: strategic, peer_resonance, cross_role, energy_identity."""
    role1 = normalize_role(p1.get("role_type"))
    role2 = normalize_role(p2.get("role_type"))

    has_investor_founder = any("Investor thesis" in r for r in reasons)
    has_fundraising = any("Fundraising" in r for r in reasons)
    has_sector_stage = any("Sector overlap" in r or "Stage alignment" in r for r in reasons)
    has_school = any("Shared completed school" in r for r in reasons)
    has_company = any("Shared company" in r for r in reasons)
    has_interest = any("Shared interest" in r for r in reasons)
    has_pivot = any("Shared pivot" in r for r in reasons)

    same_role = (role1 == role2)

    # Strategic: investor↔founder thesis, or strong sector+stage alignment across roles
    if has_investor_founder or has_fundraising:
        return "strategic"
    if not same_role and has_sector_stage and score >= 5:
        return "strategic"

    # Peer resonance: same role with shared context
    if same_role:
        return "peer_resonance"

    # Energy/identity: shared school, company, interests
    if has_school or has_company or has_interest or has_pivot:
        return "energy_identity"

    # Cross-role: different roles
    if not same_role:
        return "cross_role"

    return "cross_role"


def investor_founder_alignment_ok(p1, p2):
    """Check if investor↔founder pair passes thesis alignment rules."""
    role1 = normalize_role(p1.get("role_type"))
    role2 = normalize_role(p2.get("role_type"))

    inv, fnd = None, None
    if role1 == "Investor" and role2 == "Founder":
        inv, fnd = p1, p2
    elif role2 == "Investor" and role1 == "Founder":
        inv, fnd = p2, p1
    else:
        return True  # Not an investor-founder pair

    ip = get_investor_profile(inv)
    if not ip:
        return False  # No investor profile, can't validate

    # Low thesis clarity = broad community only
    if ip.get("thesis_clarity") == "Low":
        return False

    inv_sectors = {normalize_sector(s) for s in (ip.get("sector_focus") or []) if s}
    inv_stages = {normalize_stage(s) for s in (ip.get("stage_focus") or []) if s}
    fnd_sectors = get_sectors_set(fnd)
    fnd_stage = get_stage(fnd)

    # At least one overlap needed
    sector_overlap = bool(inv_sectors & fnd_sectors)
    stage_overlap = fnd_stage is not None and fnd_stage in inv_stages
    fnd_fundraising = fnd.get("is_fundraising") is True

    return sector_overlap or stage_overlap or fnd_fundraising


def allocate_matches(profiles, scores_matrix, hard_avoids):
    """
    Allocate exactly 4 matches per attendee using greedy allocation.
    Returns dict: attendee_index -> list of (match_index, match_type, score, reasons)
    """
    n = len(profiles)
    # Track assignments
    assignments = defaultdict(list)  # attendee_idx -> [(match_idx, type, score, reasons)]
    recommendation_count = defaultdict(int)  # how many times each person is recommended

    # Precompute sorted candidate lists for each person
    candidate_lists = {}
    for i in range(n):
        candidates = []
        for j in range(n):
            if i == j:
                continue
            if (i, j) in hard_avoids or (j, i) in hard_avoids:
                continue
            score, reasons = scores_matrix[i][j]
            if score > 0:
                match_type = classify_match_type(profiles[i], profiles[j], score, reasons)
                candidates.append((j, match_type, score, reasons))
        # Sort by score descending
        candidates.sort(key=lambda x: -x[2])
        candidate_lists[i] = candidates

    # Track which match types each person has been assigned
    assigned_types = defaultdict(set)
    # Track school-based matches per person
    school_match_count = defaultdict(int)

    def can_assign(person_idx, match_idx, match_type):
        """Check if this assignment is valid."""
        # Already matched?
        if any(m[0] == match_idx for m in assignments[person_idx]):
            return False
        # Hard cap
        if recommendation_count[match_idx] >= HARD_CAP:
            return False
        # School constraint: at most 1 school-strengthened match
        if shared_completed_school(profiles[person_idx], profiles[match_idx]):
            if school_match_count[person_idx] >= 1:
                return False
        # Investor↔founder alignment
        role_p = normalize_role(profiles[person_idx].get("role_type"))
        role_m = normalize_role(profiles[match_idx].get("role_type"))
        if (role_p == "Investor" and role_m == "Founder") or (
            role_p == "Founder" and role_m == "Investor"
        ):
            if not investor_founder_alignment_ok(
                profiles[person_idx], profiles[match_idx]
            ):
                return False
        return True

    def assign(person_idx, match_idx, match_type, score, reasons):
        """Record an assignment."""
        assignments[person_idx].append((match_idx, match_type, score, reasons))
        recommendation_count[match_idx] += 1
        assigned_types[person_idx].add(match_type)
        if shared_completed_school(profiles[person_idx], profiles[match_idx]):
            school_match_count[person_idx] += 1

    # Desired match type order for diversity
    desired_types = ["strategic", "peer_resonance", "cross_role", "energy_identity"]

    # Phase 1: Give everyone their best match
    for i in range(n):
        for j, mt, score, reasons in candidate_lists[i]:
            if can_assign(i, j, mt):
                assign(i, j, mt, score, reasons)
                break

    # Phase 2-4: Fill remaining slots, preferring type diversity
    for slot in range(1, MATCHES_PER_PERSON):
        for i in range(n):
            if len(assignments[i]) > slot:
                continue  # Already has enough

            # Determine which types we still need
            have_types = assigned_types[i]
            needed_types = [t for t in desired_types if t not in have_types]

            assigned = False

            # First try to fill a needed type (prefer soft cap)
            for needed_type in needed_types:
                # Prefer candidates under soft cap
                for prefer_soft in [True, False]:
                    for j, mt, score, reasons in candidate_lists[i]:
                        if not can_assign(i, j, mt):
                            continue
                        if prefer_soft and recommendation_count[j] >= SOFT_CAP:
                            continue
                        # Can we classify this as the needed type?
                        actual_type = classify_match_type(
                            profiles[i], profiles[j], score, reasons
                        )
                        if actual_type == needed_type:
                            assign(i, j, actual_type, score, reasons)
                            assigned = True
                            break
                    if assigned:
                        break
                if assigned:
                    break

            # If no needed type found, take best available
            if not assigned:
                for prefer_soft in [True, False]:
                    for j, mt, score, reasons in candidate_lists[i]:
                        if not can_assign(i, j, mt):
                            continue
                        if prefer_soft and recommendation_count[j] >= SOFT_CAP:
                            continue
                        actual_type = classify_match_type(
                            profiles[i], profiles[j], score, reasons
                        )
                        assign(i, j, actual_type, score, reasons)
                        assigned = True
                        break
                    if assigned:
                        break

    # Phase 5 (fallback): Relax constraints to ensure everyone has exactly 4
    for i in range(n):
        while len(assignments[i]) < MATCHES_PER_PERSON:
            assigned = False
            for j in range(n):
                if i == j:
                    continue
                if any(m[0] == j for m in assignments[i]):
                    continue
                if (i, j) in hard_avoids or (j, i) in hard_avoids:
                    continue
                score, reasons = scores_matrix[i][j]
                match_type = classify_match_type(
                    profiles[i], profiles[j], score, reasons
                )
                # Relaxed: ignore caps
                assign(i, j, match_type, score, reasons)
                assigned = True
                break
            if not assigned:
                # Absolute fallback: pick anyone not already matched
                for j in range(n):
                    if i == j:
                        continue
                    if any(m[0] == j for m in assignments[i]):
                        continue
                    assign(i, j, "cross_role", 0, ["Fallback match"])
                    assigned = True
                    break
            if not assigned:
                print(f"  WARNING: Could not find 4 matches for {profiles[i].get('name', i)}")
                break

    # Exploring protection: ensure each Exploring person has ≥1 strategic + ≥1 connector
    for i in range(n):
        role = normalize_role(profiles[i].get("role_type"))
        if role != "Exploring":
            continue
        types_assigned = [m[1] for m in assignments[i]]
        has_strategic = "strategic" in types_assigned
        has_connector = "energy_identity" in types_assigned or "cross_role" in types_assigned

        if not has_strategic and len(assignments[i]) >= MATCHES_PER_PERSON:
            # Try to swap the lowest-scoring non-strategic match for a strategic one
            worst_idx = min(
                range(len(assignments[i])),
                key=lambda x: assignments[i][x][2]
                if assignments[i][x][1] != "strategic"
                else float("inf"),
            )
            if assignments[i][worst_idx][1] != "strategic":
                old_match = assignments[i][worst_idx]
                recommendation_count[old_match[0]] -= 1
                # Find a strategic candidate
                for j, mt, score, reasons in candidate_lists[i]:
                    actual_type = classify_match_type(
                        profiles[i], profiles[j], score, reasons
                    )
                    if actual_type == "strategic" and not any(
                        m[0] == j for m in assignments[i]
                    ):
                        assignments[i][worst_idx] = (j, "strategic", score, reasons)
                        recommendation_count[j] += 1
                        break

    return assignments, recommendation_count


def validate(profiles, assignments, recommendation_count, hard_avoids):
    """Validate all constraints. Returns (passed, issues)."""
    issues = []
    n = len(profiles)

    # 1. Every attendee has exactly 4 matches
    for i in range(n):
        count = len(assignments[i])
        if count != MATCHES_PER_PERSON:
            issues.append(f"{profiles[i].get('name', i)} has {count} matches (expected {MATCHES_PER_PERSON})")

    # 2. No attendee exceeds hard cap
    for i in range(n):
        if recommendation_count[i] > HARD_CAP:
            issues.append(
                f"{profiles[i].get('name', i)} recommended {recommendation_count[i]} times (hard cap: {HARD_CAP})"
            )

    # 3. Role averages within 0.5 tolerance
    role_counts = defaultdict(list)
    for i in range(n):
        role = normalize_role(profiles[i].get("role_type"))
        role_counts[role].append(recommendation_count[i])
    role_avgs = {}
    for role, counts in role_counts.items():
        role_avgs[role] = sum(counts) / len(counts) if counts else 0
    if role_avgs:
        max_avg = max(role_avgs.values())
        min_avg = min(role_avgs.values())
        if max_avg - min_avg > 0.5:
            issues.append(
                f"Role average imbalance: {dict(role_avgs)} (diff: {max_avg - min_avg:.2f}, max: 0.5)"
            )

    # 4. All investor↔founder matches pass thesis alignment
    for i in range(n):
        for match_idx, match_type, score, reasons in assignments[i]:
            role_i = normalize_role(profiles[i].get("role_type"))
            role_m = normalize_role(profiles[match_idx].get("role_type"))
            if (role_i == "Investor" and role_m == "Founder") or (
                role_i == "Founder" and role_m == "Investor"
            ):
                if not investor_founder_alignment_ok(profiles[i], profiles[match_idx]):
                    issues.append(
                        f"Investor-founder mismatch: {profiles[i].get('name')} ↔ {profiles[match_idx].get('name')}"
                    )

    # 5. Exploring protection
    for i in range(n):
        role = normalize_role(profiles[i].get("role_type"))
        if role != "Exploring":
            continue
        types = [m[1] for m in assignments[i]]
        if "strategic" not in types:
            issues.append(f"Exploring attendee {profiles[i].get('name')} missing strategic match")
        if "energy_identity" not in types and "cross_role" not in types:
            issues.append(f"Exploring attendee {profiles[i].get('name')} missing connector match")

    # 6. School constraint (at most 1 school-strengthened match)
    for i in range(n):
        school_matches = 0
        for match_idx, match_type, score, reasons in assignments[i]:
            if shared_completed_school(profiles[i], profiles[match_idx]):
                school_matches += 1
        if school_matches > 1:
            issues.append(
                f"{profiles[i].get('name')} has {school_matches} school-strengthened matches (max 1)"
            )

    # 7. No hard-avoid pairs matched
    for i in range(n):
        for match_idx, match_type, score, reasons in assignments[i]:
            if (i, match_idx) in hard_avoids or (match_idx, i) in hard_avoids:
                issues.append(
                    f"Hard-avoid pair matched: {profiles[i].get('name')} ↔ {profiles[match_idx].get('name')}"
                )

    passed = len(issues) == 0
    return passed, issues


def main():
    parser = argparse.ArgumentParser(description="Deterministic matchmaking (zero LLM)")
    parser.add_argument("--input", required=True, help="Path to profiles.json")
    parser.add_argument("--output", required=True, help="Output path for matches.json")
    args = parser.parse_args()

    # Load profiles
    print(f"Loading profiles: {args.input}")
    with open(args.input, "r") as f:
        profiles = json.load(f)

    # Filter out extraction errors
    valid_profiles = []
    skipped = 0
    for p in profiles:
        if "_extraction_error" in p:
            print(f"  Skipping {p.get('name', '?')}: extraction error")
            skipped += 1
        else:
            valid_profiles.append(p)

    profiles = valid_profiles
    n = len(profiles)
    print(f"Processing {n} valid profiles ({skipped} skipped due to errors)")

    # Role distribution
    role_dist = defaultdict(int)
    for p in profiles:
        role_dist[normalize_role(p.get("role_type"))] += 1
    print(f"Role distribution: {dict(role_dist)}")

    # Phase 1: Compute all pairwise scores
    print("\nPhase 1: Computing pairwise scores...")
    scores_matrix = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                scores_matrix[i][j] = (0, [])
            elif j > i:
                score, reasons = compute_pairwise_score(profiles[i], profiles[j])
                scores_matrix[i][j] = (score, reasons)
                scores_matrix[j][i] = (score, reasons)

    total_pairs = n * (n - 1) // 2
    nonzero = sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if scores_matrix[i][j][0] > 0
    )
    print(f"  {total_pairs} pairs computed, {nonzero} with positive scores")

    # Phase 2: Identify hard avoids
    print("\nPhase 2: Identifying hard avoids...")
    hard_avoids = set()
    for i in range(n):
        for j in range(i + 1, n):
            if check_hard_avoid(profiles[i], profiles[j]):
                hard_avoids.add((i, j))
    print(f"  {len(hard_avoids)} hard-avoid pairs found")

    # Phase 3: Allocate matches
    print("\nPhase 3: Allocating matches...")
    assignments, recommendation_count = allocate_matches(
        profiles, scores_matrix, hard_avoids
    )

    # Print allocation stats
    rec_counts = [recommendation_count[i] for i in range(n)]
    print(f"  Recommendation counts: min={min(rec_counts)}, max={max(rec_counts)}, "
          f"avg={sum(rec_counts)/len(rec_counts):.1f}")
    over_soft = sum(1 for c in rec_counts if c > SOFT_CAP)
    print(f"  Over soft cap ({SOFT_CAP}): {over_soft} people")

    # Phase 4: Validate
    print("\nPhase 4: Validating...")
    passed, issues = validate(profiles, assignments, recommendation_count, hard_avoids)
    if passed:
        print("  \u2713 All validation checks passed!")
    else:
        print(f"  \u2717 {len(issues)} validation issues found:")
        for issue in issues:
            print(f"    - {issue}")

    # Build output
    print("\nBuilding output...")
    output = {
        "metadata": {
            "total_attendees": n,
            "skipped_errors": skipped,
            "hard_avoids": len(hard_avoids),
            "validation_passed": passed,
            "validation_issues": issues,
        },
        "matches": [],
    }

    for i in range(n):
        attendee = {
            "attendee_index": i,
            "attendee_name": profiles[i].get("name", ""),
            "attendee_email": profiles[i].get("_email", ""),
            "attendee_role": normalize_role(profiles[i].get("role_type")),
            "matches": [],
        }
        for match_idx, match_type, score, reasons in assignments[i]:
            match_entry = {
                "match_index": match_idx,
                "match_name": profiles[match_idx].get("name", ""),
                "match_email": profiles[match_idx].get("_email", ""),
                "match_linkedin_url": profiles[match_idx].get("_linkedin_url", ""),
                "match_role": normalize_role(profiles[match_idx].get("role_type")),
                "match_type": match_type,
                "score": score,
                "reasons": reasons,
            }
            attendee["matches"].append(match_entry)
        output["matches"].append(attendee)

    # Save
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved {n} attendee matches to {args.output}")


if __name__ == "__main__":
    main()
