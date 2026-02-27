#!/usr/bin/env python3
"""
generate.py — LLM per-attendee email/fun-fact writing

Reads profiles.json, matches.json, and the original spreadsheet.
Calls Claude API once per attendee with ONLY their profile + 4 matched profiles.
Generates personalized email drafts and fun facts.
Writes final_output.xlsx.

Usage:
    python generate.py --profiles profiles.json --matches matches.json \
                       --input Final_Guest_List.xlsx --output final_output.xlsx
    python generate.py --profiles profiles.json --matches matches.json \
                       --input Final_Guest_List.xlsx --output final_output.xlsx \
                       --start 5 --limit 10
"""

import argparse
import json
import os
import sys
import time

import anthropic
import pandas as pd

DEFAULT_MODEL = "claude-sonnet-4-5-20250514"

SYSTEM_PROMPT = """You are Anna and Isa, co-founders of Circe. Write warm, concise, personalized content based ONLY on the data provided. No invented facts. No generic phrasing.

Rules:
- Every personal detail must come from the attendee's profile data (LinkedIn text, survey fields, or extracted profile).
- Every "ask them about" must reference a SPECIFIC fact from the match's profile (a project, company, achievement, school, etc.).
- No vague language like "you two have so much in common" or "you'll really click."
- Fun facts must be concrete and factual — no personality inference or emotional framing.
- Return ONLY valid JSON, no markdown formatting, no explanation."""


def build_generation_prompt(attendee_profile, match_profiles, match_data):
    """Build the per-attendee prompt for email + fun facts generation."""
    first_name = attendee_profile.get("first_name", attendee_profile.get("name", "").split()[0] if attendee_profile.get("name") else "")

    prompt = f"""Generate a personalized matchmaking email and fun facts for this attendee.

## Attendee Profile
```json
{json.dumps(attendee_profile, indent=2, ensure_ascii=False)}
```

## Their 4 Matches

"""
    for idx, (match_profile, match_info) in enumerate(zip(match_profiles, match_data), 1):
        linkedin_url = match_profile.get("_linkedin_url", "") or match_info.get("match_linkedin_url", "") or ""
        prompt += f"""### Match {idx}: {match_info.get('match_name', '')}
- LinkedIn URL: {linkedin_url}
- Match Type: {match_info.get('match_type', '')}
- Match Score: {match_info.get('score', 0)}
- Match Reasons: {', '.join(match_info.get('reasons', []))}

Profile:
```json
{json.dumps(match_profile, indent=2, ensure_ascii=False)}
```

"""

    prompt += f"""## Required Output

Return a JSON object with two fields:

### email_draft
Write the email in this EXACT format:

Hi {first_name}! We are so excited you made it today. We can't wait for you to tell us a bit about [personal detail grounded in their profile data]. We think that while you're here, you'll benefit from talking to these people:

[Match 1 Full Name] ([LinkedIn URL]) — You'll enjoy talking to them because [rationale grounded in match reasons + profiles]. Make sure to ask them about [specific factual hook from match's profile].

[Match 2 Full Name] ([LinkedIn URL]) — You'll enjoy talking to them because [rationale]. Make sure to ask them about [specific factual hook].

[Match 3 Full Name] ([LinkedIn URL]) — You'll enjoy talking to them because [rationale]. Make sure to ask them about [specific factual hook].

[Match 4 Full Name] ([LinkedIn URL]) — You'll enjoy talking to them because [rationale]. Make sure to ask them about [specific factual hook].

### fun_facts
3-5 bullet points about the ATTENDEE (not their matches):
- Start each with "- "
- Max 16 words each
- Fact-based only, no speculation, no personality inference
- Highlight: notable companies, competitions, fellowships, completed schools, career pivots, concrete achievements
- Avoid: generic job restatement, emotional framing

Return ONLY the JSON object:
{{"email_draft": "the full email text", "fun_facts": "- fact 1\\n- fact 2\\n- fact 3"}}"""

    return prompt


def load_spreadsheet(path):
    """Load and clean the guest list spreadsheet."""
    df = pd.read_excel(path, header=1)

    # Drop the empty first column (index 0)
    if df.columns[0] == "Unnamed: 0" or pd.isna(df.columns[0]):
        df = df.drop(df.columns[0], axis=1)
    elif str(df.columns[0]).startswith("Unnamed"):
        df = df.drop(df.columns[0], axis=1)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate matchmaking emails and fun facts using Claude API"
    )
    parser.add_argument("--profiles", required=True, help="Path to profiles.json")
    parser.add_argument("--matches", required=True, help="Path to matches.json")
    parser.add_argument("--input", required=True, help="Path to Final_Guest_List.xlsx")
    parser.add_argument("--output", required=True, help="Output path for final_output.xlsx")
    parser.add_argument("--start", type=int, default=0, help="Resume from attendee N")
    parser.add_argument("--limit", type=int, default=None, help="Process only N attendees")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model to use")
    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load data
    print(f"Loading profiles: {args.profiles}")
    with open(args.profiles, "r") as f:
        profiles = json.load(f)
    print(f"  {len(profiles)} profiles loaded")

    print(f"Loading matches: {args.matches}")
    with open(args.matches, "r") as f:
        matches_data = json.load(f)
    matches_list = matches_data["matches"]
    print(f"  {len(matches_list)} attendee match sets loaded")

    print(f"Loading spreadsheet: {args.input}")
    df = load_spreadsheet(args.input)
    print(f"  {len(df)} rows in spreadsheet")

    # Build profile index by _row_index for quick lookup
    profile_by_row = {}
    for p in profiles:
        if "_row_index" in p:
            profile_by_row[p["_row_index"]] = p

    # Build profile index by match_index (position in valid profiles list)
    # The match_index in matches.json refers to position in the valid profiles array
    valid_profiles = [p for p in profiles if "_extraction_error" not in p]
    profile_by_match_idx = {i: p for i, p in enumerate(valid_profiles)}

    # Load existing generated content if resuming
    generated = {}
    intermediate_path = args.output.replace(".xlsx", "_intermediate.json")
    if args.start > 0 and os.path.exists(intermediate_path):
        print(f"Resuming from attendee {args.start}, loading existing intermediate file")
        with open(intermediate_path, "r") as f:
            generated = json.load(f)
        print(f"  {len(generated)} existing generations loaded")

    # Determine range
    start = args.start
    end = len(matches_list)
    if args.limit is not None:
        end = min(start + args.limit, len(matches_list))

    total = end - start
    print(f"\nProcessing attendees {start} to {end - 1} ({total} attendees)")
    print(f"Using model: {args.model}")
    print()

    for idx in range(start, end):
        attendee_data = matches_list[idx]
        attendee_name = attendee_data["attendee_name"]
        attendee_index = attendee_data["attendee_index"]

        progress = idx - start + 1
        print(f"[{progress}/{total}] Generating for: {attendee_name}...", end=" ", flush=True)

        # Get attendee profile
        attendee_profile = profile_by_match_idx.get(attendee_index)
        if not attendee_profile:
            print("\u2717 (profile not found, skipping)")
            continue

        # Get match profiles
        match_profiles = []
        match_infos = attendee_data["matches"]
        for m in match_infos:
            mp = profile_by_match_idx.get(m["match_index"])
            if mp:
                match_profiles.append(mp)
            else:
                # Fallback: create minimal profile from match info
                match_profiles.append({
                    "name": m.get("match_name", ""),
                    "role_type": m.get("match_role", ""),
                    "_linkedin_url": m.get("match_linkedin_url", ""),
                    "_email": m.get("match_email", ""),
                })

        # Build prompt
        prompt = build_generation_prompt(attendee_profile, match_profiles, match_infos)

        # Retry loop
        success = False
        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=args.model,
                    max_tokens=2000,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )

                text = response.content[0].text.strip()

                # Remove markdown code fences if present
                if text.startswith("```"):
                    lines = text.split("\n")
                    lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    text = "\n".join(lines)

                result = json.loads(text)

                if "email_draft" not in result or "fun_facts" not in result:
                    raise ValueError("Missing email_draft or fun_facts in response")

                # Store by row_index for merging
                row_idx = attendee_profile.get("_row_index")
                if row_idx is not None:
                    generated[str(row_idx)] = {
                        "email_draft": result["email_draft"],
                        "fun_facts": result["fun_facts"],
                        "name": attendee_name,
                    }

                success = True
                # Truncate for display
                email_preview = result["email_draft"][:60].replace("\n", " ")
                print(f"\u2713 ({email_preview}...)")
                break

            except json.JSONDecodeError as e:
                print(f"\n  JSON parse error (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(1)
            except anthropic.APIError as e:
                print(f"\n  API error (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
            except Exception as e:
                print(f"\n  Error (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(1)

        if not success:
            print("\u2717 (failed after 3 attempts)")

        # Checkpoint every 10
        if progress % 10 == 0:
            with open(intermediate_path, "w") as f:
                json.dump(generated, f, indent=2, ensure_ascii=False)
            print(f"  [Checkpoint saved: {len(generated)} generations]")

        # Rate limit
        time.sleep(0.3)

    # Save intermediate
    with open(intermediate_path, "w") as f:
        json.dump(generated, f, indent=2, ensure_ascii=False)

    # Build final output
    print(f"\nBuilding final output: {args.output}")

    # Reload original spreadsheet (full, with header at row 1)
    df_full = pd.read_excel(args.input, header=1)

    # Drop empty first column
    if df_full.columns[0] == "Unnamed: 0" or pd.isna(df_full.columns[0]):
        df_full = df_full.drop(df_full.columns[0], axis=1)
    elif str(df_full.columns[0]).startswith("Unnamed"):
        df_full = df_full.drop(df_full.columns[0], axis=1)

    # Find name column and filter to valid rows
    name_col = None
    for col in df_full.columns:
        if str(col).strip().lower() == "name":
            name_col = col
            break
    if name_col is None:
        name_col = df_full.columns[0]

    # Keep all rows but add columns
    email_drafts = []
    fun_facts_list = []

    for i in range(len(df_full)):
        row_key = str(i)
        if row_key in generated:
            email_drafts.append(generated[row_key].get("email_draft", ""))
            fun_facts_list.append(generated[row_key].get("fun_facts", ""))
        else:
            email_drafts.append("")
            fun_facts_list.append("")

    df_full["Email Draft"] = email_drafts
    df_full["Fun Facts"] = fun_facts_list

    # Save with openpyxl for better formatting
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        df_full.to_excel(writer, index=False, sheet_name="Sheet1")

        # Adjust column widths
        worksheet = writer.sheets["Sheet1"]
        for col_idx, col_name in enumerate(df_full.columns, 1):
            if col_name == "Email Draft":
                worksheet.column_dimensions[
                    worksheet.cell(row=1, column=col_idx).column_letter
                ].width = 80
            elif col_name == "Fun Facts":
                worksheet.column_dimensions[
                    worksheet.cell(row=1, column=col_idx).column_letter
                ].width = 50
            elif col_name == "LinkedIn Profile paste":
                worksheet.column_dimensions[
                    worksheet.cell(row=1, column=col_idx).column_letter
                ].width = 30
            else:
                worksheet.column_dimensions[
                    worksheet.cell(row=1, column=col_idx).column_letter
                ].width = 18

    print(f"\nDone! Saved {args.output}")
    print(f"  {len(generated)} attendees have email drafts + fun facts")

    # Cleanup intermediate
    print(f"  Intermediate file kept at: {intermediate_path}")


if __name__ == "__main__":
    main()
