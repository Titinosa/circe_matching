#!/usr/bin/env python3
"""
extract.py — LLM per-attendee extraction

Reads Final_Guest_List.xlsx, calls Claude API once per attendee to extract
a structured JSON profile from their LinkedIn text + survey fields.
Writes profiles.json.

Usage:
    python extract.py --input Final_Guest_List.xlsx --output profiles.json
    python extract.py --input Final_Guest_List.xlsx --output profiles.json --start 5 --limit 10
"""

import argparse
import json
import os
import sys
import time

import anthropic
import pandas as pd

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

EXTRACTION_SCHEMA = """{
  "name": "Full Name",
  "first_name": "First",
  "last_name": "Last",
  "role_type": "Founder | Operator | Investor | Exploring",
  "sectors": ["AI/ML", "SaaS"],
  "functional_strengths": ["Product", "Engineering"],
  "schools": [
    {
      "name": "UC Berkeley",
      "degree": "MBA | BS | MS | PhD | JD | MPH | null",
      "discipline": "Computer Science | Business | null",
      "start_year": 2018,
      "end_year": 2022,
      "completed": true
    }
  ],
  "notable_companies": ["Adobe", "Google"],
  "current_company": "Company or null",
  "current_title": "Title or null",
  "stage": "Preseed | Seed | Series A | ... | null",
  "team_size": "0-1 | 2-5 | ... | null",
  "top_of_mind": ["Building / product", "Fundraising"],
  "is_hiring": true,
  "hiring_timeline": "now | 3-6 months | no | null",
  "has_technical_needs": true,
  "is_fundraising": true,
  "investor_profile": {
    "sector_focus": ["Fintech", "AI/ML"],
    "stage_focus": ["Pre-seed", "Seed"],
    "capital_type": "VC | Angel | Family Office | null",
    "thesis_clarity": "High | Medium | Low"
  },
  "geography": "San Francisco, CA | null",
  "career_pivots": ["Finance to Tech", "Corporate to Founder"],
  "explicit_interests": ["interest from posts or about section"],
  "key_facts_for_fun_facts": [
    "Won TechCrunch Disrupt 2023",
    "Former Olympic athlete",
    "Published in Nature"
  ]
}"""

SYSTEM_PROMPT = """You are a precise data extraction assistant. Your job is to extract structured information from a person's LinkedIn profile text and survey responses.

RULES:
- ONLY extract what is explicitly present in the provided text. Do NOT infer, guess, or fabricate any information.
- Use null for fields that cannot be determined from the data.
- Use empty lists [] for list fields where no information is available.
- For sector tags, use these canonical labels ONLY: AI/ML, Fintech, Healthcare, SaaS, Consumer, Climate, Enterprise Software, Edtech, Biotech, Crypto/Web3, E-commerce, Media, Real Estate, Legal Tech, HR Tech, Dev Tools, Cybersecurity, Hardware, Robotics, Food/Bev, Marketplace, Social Impact, Defense/Gov Tech
- For functional strengths, use these labels ONLY: Product, Engineering, Sales, Marketing, Growth, Operations, Finance, Legal, Data Science, Design, Strategy, People/HR, BD/Partnerships, Fundraising, Community, Content, Research, Supply Chain, Customer Success
- investor_profile should be null unless the person's identity is Investor.
- is_fundraising should be true if "Fundraising" appears in their top_of_mind survey response.
- top_of_mind should be split from the comma-separated survey value into a list.
- key_facts_for_fun_facts: Extract 3-5 distinctive, CONCRETE facts — not generic job descriptions. Look for: competitions won, fellowships, publications, unusual career pivots, notable achievements, specific metrics mentioned, unique hobbies or accomplishments.
- For schools, set completed=true only if there is evidence they finished the program. If unclear, use completed=null.

Return ONLY valid JSON, no markdown formatting, no explanation."""


def get_user_prompt(row, linkedin_text):
    """Build the per-attendee user prompt."""
    # Gather survey fields
    name = row.get("name", "")
    first_name = row.get("first_name", "")
    last_name = row.get("last_name", "")
    identity = row.get("How do you primarily identify right now?", "")
    top_of_mind = row.get("What\u2019s top of mind for you right now?", "")
    team_size = row.get(
        "Founders: How many employees do you currently have? (Including founders)", ""
    )
    hiring = row.get("Are you hiring?", "")
    tech_needs = row.get(
        "Do you have engineering/technical needs beyond your/your teams capacity?", ""
    )
    stage = row.get("Founders: What stage is your company?", "")

    # Clean NaN values
    def clean(val):
        if pd.isna(val):
            return ""
        return str(val).strip()

    name = clean(name)
    first_name = clean(first_name)
    last_name = clean(last_name)
    identity = clean(identity)
    top_of_mind = clean(top_of_mind)
    team_size = clean(team_size)
    hiring = clean(hiring)
    tech_needs = clean(tech_needs)
    stage = clean(stage)

    # Truncate LinkedIn text if too long
    if linkedin_text and len(linkedin_text) > 15000:
        linkedin_text = linkedin_text[:15000] + "\n... [truncated]"

    prompt = f"""Extract a structured profile for this person.

## Survey Data
- Name: {name}
- First Name: {first_name}
- Last Name: {last_name}
- Identity: {identity}
- Top of Mind: {top_of_mind}
- Team Size: {team_size}
- Hiring: {hiring}
- Technical Needs: {tech_needs}
- Company Stage: {stage}

## LinkedIn Profile Text
{linkedin_text if linkedin_text else "[No LinkedIn text available — extract what you can from survey data only]"}

## Required Output Schema
Return a JSON object matching this exact schema:
{EXTRACTION_SCHEMA}

Remember:
- investor_profile should be null unless identity is "Investor"
- is_fundraising = true if "Fundraising" appears in Top of Mind
- Split top_of_mind from the comma-separated survey value into a list
- key_facts_for_fun_facts: 3-5 distinctive CONCRETE facts, not generic job descriptions
- Return ONLY the JSON object, nothing else."""

    return prompt


def find_column(df, search_term):
    """Find a column by fuzzy matching (handles curly quotes etc.)."""
    for col in df.columns:
        # Normalize quotes for comparison
        normalized = col.replace("\u2019", "'").replace("\u2018", "'")
        search_normalized = search_term.replace("\u2019", "'").replace("\u2018", "'")
        if search_normalized.lower() in normalized.lower():
            return col
    return None


def load_spreadsheet(path):
    """Load and clean the guest list spreadsheet."""
    df = pd.read_excel(path, header=1)

    # Drop the empty first column (index 0)
    if df.columns[0] == "Unnamed: 0" or pd.isna(df.columns[0]):
        df = df.drop(df.columns[0], axis=1)
    elif str(df.columns[0]).startswith("Unnamed"):
        df = df.drop(df.columns[0], axis=1)

    # Filter to rows with non-null name
    name_col = find_column(df, "name")
    if name_col is None:
        # Try exact match
        if "name" in df.columns:
            name_col = "name"
        else:
            raise ValueError(f"Cannot find 'name' column. Columns: {list(df.columns)}")

    df = df[df[name_col].notna()].reset_index(drop=True)

    return df


def extract_profile(client, model, row, linkedin_text):
    """Call Claude API to extract a profile for one attendee."""
    user_prompt = get_user_prompt(row, linkedin_text)

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Extract text from response
    text = response.content[0].text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    profile = json.loads(text)
    return profile


def attach_raw_metadata(profile, row, row_index, df):
    """Attach raw spreadsheet fields directly (not through LLM)."""
    def clean(val):
        if pd.isna(val):
            return None
        return str(val).strip()

    email_col = find_column(df, "email")
    linkedin_url_col = find_column(df, "LinkedIn profile?")
    identity_col = find_column(df, "How do you primarily identify")
    top_of_mind_col = find_column(df, "top of mind")
    team_size_col = find_column(df, "How many employees")
    hiring_col = find_column(df, "Are you hiring")
    tech_needs_col = find_column(df, "engineering/technical needs")
    stage_col = find_column(df, "What stage is your company")

    profile["_email"] = clean(row.get(email_col, None)) if email_col else None
    profile["_linkedin_url"] = (
        clean(row.get(linkedin_url_col, None)) if linkedin_url_col else None
    )
    profile["_row_index"] = row_index
    profile["_identity_raw"] = (
        clean(row.get(identity_col, None)) if identity_col else None
    )
    profile["_top_of_mind_raw"] = (
        clean(row.get(top_of_mind_col, None)) if top_of_mind_col else None
    )
    profile["_team_size_raw"] = (
        clean(row.get(team_size_col, None)) if team_size_col else None
    )
    profile["_hiring_raw"] = (
        clean(row.get(hiring_col, None)) if hiring_col else None
    )
    profile["_tech_needs_raw"] = (
        clean(row.get(tech_needs_col, None)) if tech_needs_col else None
    )
    profile["_stage_raw"] = (
        clean(row.get(stage_col, None)) if stage_col else None
    )

    return profile


def save_profiles(profiles, output_path):
    """Save profiles list to JSON."""
    with open(output_path, "w") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Extract attendee profiles using Claude API")
    parser.add_argument("--input", required=True, help="Path to Final_Guest_List.xlsx")
    parser.add_argument("--output", required=True, help="Output path for profiles.json")
    parser.add_argument("--start", type=int, default=0, help="Resume from row N")
    parser.add_argument("--limit", type=int, default=None, help="Process only N rows")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model to use")
    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load spreadsheet
    print(f"Loading spreadsheet: {args.input}")
    df = load_spreadsheet(args.input)
    print(f"Found {len(df)} attendees")

    # Find LinkedIn text column
    linkedin_col = find_column(df, "LinkedIn Profile paste")
    if linkedin_col is None:
        print("WARNING: Cannot find LinkedIn Profile paste column")
        linkedin_col = None

    # Load existing profiles if resuming
    profiles = []
    if args.start > 0 and os.path.exists(args.output):
        print(f"Resuming from row {args.start}, loading existing {args.output}")
        with open(args.output, "r") as f:
            profiles = json.load(f)
        print(f"Loaded {len(profiles)} existing profiles")

    # Determine range
    start = args.start
    end = len(df)
    if args.limit is not None:
        end = min(start + args.limit, len(df))

    total = end - start
    print(f"Processing rows {start} to {end - 1} ({total} attendees)")
    print(f"Using model: {args.model}")
    print()

    for i in range(start, end):
        row = df.iloc[i]
        name = row.get("name", f"Row {i}")
        if pd.isna(name):
            name = f"Row {i}"

        progress = i - start + 1
        print(f"[{progress}/{total}] Extracting: {name}...", end=" ", flush=True)

        # Get LinkedIn text
        linkedin_text = None
        if linkedin_col is not None:
            raw = row.get(linkedin_col, None)
            if not pd.isna(raw):
                linkedin_text = str(raw).strip()

        # Retry loop
        success = False
        for attempt in range(3):
            try:
                profile = extract_profile(client, args.model, row, linkedin_text)
                profile = attach_raw_metadata(profile, row, i, df)
                profiles.append(profile)
                success = True

                # Print summary
                role = profile.get("role_type", "?")
                sectors = ", ".join(profile.get("sectors", [])[:3])
                strengths = ", ".join(profile.get("functional_strengths", [])[:2])
                summary = f"{role}"
                if sectors:
                    summary += f" | {sectors}"
                if strengths:
                    summary += f" | {strengths}"
                print(f"\u2713 ({summary})")
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
            # Save partial profile with error
            error_profile = {
                "name": str(name),
                "_row_index": i,
                "_extraction_error": f"Failed after 3 attempts",
                "_email": None,
                "_linkedin_url": None,
            }
            # Try to attach raw metadata
            try:
                error_profile = attach_raw_metadata(error_profile, row, i, df)
            except Exception:
                pass
            profiles.append(error_profile)
            print(f"\u2717 (saved partial profile with error)")

        # Checkpoint every 10 profiles
        if progress % 10 == 0:
            save_profiles(profiles, args.output)
            print(f"  [Checkpoint saved: {len(profiles)} profiles]")

        # Rate limit politeness
        time.sleep(0.3)

    # Final save
    save_profiles(profiles, args.output)
    print()
    print(f"Done! Saved {len(profiles)} profiles to {args.output}")

    # Summary
    errors = sum(1 for p in profiles if "_extraction_error" in p)
    if errors:
        print(f"  ({errors} profiles had extraction errors)")


if __name__ == "__main__":
    main()
