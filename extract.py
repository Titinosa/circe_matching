#!/usr/bin/env python3
"""
extract.py — LLM per-attendee extraction

Reads New Event Guestlist.xls, calls Claude API once per attendee to extract
a structured JSON profile from their LinkedIn text + survey fields.
Writes profiles.json.

Usage:
    python extract.py --input "New Event Guestlist.xls" --output profiles.json
    python extract.py --input "New Event Guestlist.xls" --output profiles.json --start 5 --limit 10
"""

import argparse
import json
import os
import sys
import time

import anthropic
import pandas as pd

DEFAULT_MODEL = "claude-sonnet-4-5-20250514"

EXTRACTION_SCHEMA = """{
  "name": "Full Name",
  "first_name": "First",
  "last_name": "Last",
  "role_type": "Founder | Operator | Investor | Exploring",
  "role_detail": "Investor (Early Stage) | Partnerships | Engineering | null",
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
  "stage": "Ideating | Pre-Seed | Seed | Series A | Series B+ | Later-stage | Just Curious | null",
  "event_goal": "Peer Connection | Fundraising | Hiring | Mentorship | Prospecting | Pipeline | Paying it Forward | Just Curious | null",
  "superpower": "The Zero-to-One | The Scaler | The Connector | The Operator | The Storyteller | The Architect | Other | null",
  "desired_connections": "The Mirror | The Collaborator | The Mentor | The Window | Investors | Engineers/Operators | Co-founder | null",
  "bullish_trend": "short summary of the trend or null",
  "bullish_trend_tags": ["AI", "community", "fintech"],
  "community_hopes": "short summary of what they hope to get or null",
  "community_hopes_tags": ["mentorship", "networking", "collaboration"],
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
    "Published in Nature",
    "Built a product used by 50k users",
    "Started career in investment banking at Goldman Sachs",
    "Co-founded a nonprofit for women in STEM",
    "Holds a patent in distributed systems",
    "Speaker at Web Summit 2024",
    "Ran a marathon on every continent",
    "Former YC batch W23"
  ]
}"""

SYSTEM_PROMPT = """You are a precise data extraction assistant. Your job is to extract structured information from a person's LinkedIn profile text and survey responses.

RULES:
- ONLY extract what is explicitly present in the provided text. Do NOT infer, guess, or fabricate any information.
- Use null for fields that cannot be determined from the data.
- Use empty lists [] for list fields where no information is available.
- For sector tags, use these canonical labels ONLY: AI/ML, Fintech, Healthcare, SaaS, Consumer, Climate, Enterprise Software, Edtech, Biotech, Crypto/Web3, E-commerce, Media, Real Estate, Legal Tech, HR Tech, Dev Tools, Cybersecurity, Hardware, Robotics, Food/Bev, Marketplace, Social Impact, Defense/Gov Tech, Space/Defense Tech, HealthTech
- For functional strengths, use these labels ONLY: Product, Engineering, Sales, Marketing, Growth, Operations, Finance, Legal, Data Science, Design, Strategy, People/HR, BD/Partnerships, Fundraising, Community, Content, Research, Supply Chain, Customer Success
- investor_profile should be null unless the person's identity is Investor.
- is_fundraising should be true if the event goal mentions "Fundraising" or looking for investors/partners.
- event_goal: Normalize from the survey answer. Use one of: Peer Connection, Fundraising, Hiring, Mentorship, Prospecting, Pipeline, Paying it Forward, Just Curious. Use null if unclear.
- superpower: Normalize from survey. Use one of: The Zero-to-One, The Scaler, The Connector, The Operator, The Storyteller, The Architect, Other. Use null if unclear.
- desired_connections: Normalize from survey. Use one of: The Mirror, The Collaborator, The Mentor, The Window, Investors, Engineers/Operators, Co-founder. Use null if unclear.
- bullish_trend: Summarize their stated trend in 10 words or fewer. Use null if empty.
- bullish_trend_tags: Extract 1-4 canonical topic tags from their trend text (e.g., AI, community, fintech, health, real estate, defense). Use [].
- community_hopes: Summarize what they want from the community in 10 words or fewer. Use null if empty.
- community_hopes_tags: Extract 1-4 canonical tags (e.g., mentorship, networking, collaboration, support, co-founders, investors). Use [].
- key_facts_for_fun_facts: Extract 8-10 distinctive, CONCRETE facts — not generic job descriptions. Look for: competitions won, fellowships, publications, unusual career pivots, notable achievements, specific metrics mentioned, unique hobbies or accomplishments, notable companies worked at, schools attended, interesting projects. Try harder to find more facts from LinkedIn text.
- For schools, set completed=true only if there is evidence they finished the program. If unclear, use completed=null.
- role_detail: Capture any more specific role/function description if available (e.g., "Investor (Early Stage)", "Partnerships", "Engineering"). Use null if only the broad role is available.

Return ONLY valid JSON, no markdown formatting, no explanation."""


def get_user_prompt(row, linkedin_text):
    """Build the per-attendee user prompt."""
    # Gather survey fields
    name = row.get("name", "")
    first_name = row.get("first_name", "")
    last_name = row.get("last_name", "")
    # Role from the explicit role/function column
    identity = row.get("Which of these Role/Function apply to you (past or present!)", "")
    # Detailed role/function from LINKEDIN PROFILE2
    role_detail = row.get("LINKEDIN PROFILE2", "")
    stage = row.get("If you're at a startup, describe your stage:", "")
    sector = row.get("Industry/Sector", "")
    event_goal = row.get("What's your primary goal for today's event?", "")
    superpower = row.get("What is your Superpower", "")
    desired_connections = row.get(
        "What kind of people would be most valuable for you to meet?", ""
    )
    bullish_trend = row.get(
        "What is one trend (in tech or SF) that you are genuinely bullish on?", ""
    )
    community_hopes = row.get(
        "We are so glad you found Circe. What are you hoping to get out of this community?", ""
    )

    # Clean NaN values
    def clean(val):
        if pd.isna(val):
            return ""
        return str(val).strip()

    name = clean(name)
    first_name = clean(first_name)
    last_name = clean(last_name)
    identity = clean(identity)
    role_detail = clean(role_detail)
    stage = clean(stage)
    sector = clean(sector)
    event_goal = clean(event_goal)
    superpower = clean(superpower)
    desired_connections = clean(desired_connections)
    bullish_trend = clean(bullish_trend)
    community_hopes = clean(community_hopes)

    # Truncate LinkedIn text if too long
    if linkedin_text and len(linkedin_text) > 15000:
        linkedin_text = linkedin_text[:15000] + "\n... [truncated]"

    prompt = f"""Extract a structured profile for this person.

## Survey Data
- Name: {name}
- First Name: {first_name}
- Last Name: {last_name}
- Identity: {identity}
- Role/Function Detail: {role_detail}
- Company Stage: {stage}
- Industry/Sector: {sector}
- Primary Goal for Event: {event_goal}
- Superpower: {superpower}
- Most Valuable Connections: {desired_connections}
- Bullish Trend: {bullish_trend}
- Community Hopes: {community_hopes}

## LinkedIn Profile Text
{linkedin_text if linkedin_text else "[No LinkedIn text available — extract what you can from survey data only]"}

## Required Output Schema
Return a JSON object matching this exact schema:
{EXTRACTION_SCHEMA}

Remember:
- investor_profile should be null unless identity is "Investor"
- is_fundraising = true if event goal mentions "Fundraising" or looking for investors
- event_goal, superpower, desired_connections: normalize from the survey values to canonical labels
- bullish_trend: summarize in 10 words or fewer; bullish_trend_tags: 1-4 topic tags
- community_hopes: summarize in 10 words or fewer; community_hopes_tags: 1-4 tags
- key_facts_for_fun_facts: 8-10 distinctive CONCRETE facts, not generic job descriptions
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
    # Try header at row 0 first (new format), fall back to row 1 (old format)
    df = pd.read_excel(path, header=0)

    # Check if the first column is 'name' or similar; if not, try header=1
    has_name = any(str(c).strip().lower() == "name" for c in df.columns)
    if not has_name:
        df = pd.read_excel(path, header=1)

    # Drop the empty first column (index 0) if present
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
    # In the new guestlist, LinkedIn URL is in "Which of these best describes you right now:"
    linkedin_url_col = find_column(df, "best describes you right now")
    if linkedin_url_col is None:
        linkedin_url_col = find_column(df, "LinkedIn profile?")
    identity_col = find_column(df, "Role/Function apply")
    if identity_col is None:
        identity_col = find_column(df, "How do you primarily identify")
    stage_col = find_column(df, "describe your stage")
    if stage_col is None:
        stage_col = find_column(df, "What stage is your company")
    sector_col = find_column(df, "Industry/Sector")
    event_goal_col = find_column(df, "primary goal for today")
    superpower_col = find_column(df, "Superpower")
    desired_conn_col = find_column(df, "most valuable for you to meet")
    bullish_col = find_column(df, "genuinely bullish on")
    community_col = find_column(df, "hoping to get out of this community")

    profile["_email"] = clean(row.get(email_col, None)) if email_col else None
    profile["_linkedin_url"] = (
        clean(row.get(linkedin_url_col, None)) if linkedin_url_col else None
    )
    profile["_row_index"] = row_index
    profile["_identity_raw"] = (
        clean(row.get(identity_col, None)) if identity_col else None
    )
    profile["_stage_raw"] = (
        clean(row.get(stage_col, None)) if stage_col else None
    )
    profile["_sector_raw"] = (
        clean(row.get(sector_col, None)) if sector_col else None
    )
    profile["_event_goal_raw"] = (
        clean(row.get(event_goal_col, None)) if event_goal_col else None
    )
    profile["_superpower_raw"] = (
        clean(row.get(superpower_col, None)) if superpower_col else None
    )
    profile["_desired_connections_raw"] = (
        clean(row.get(desired_conn_col, None)) if desired_conn_col else None
    )
    profile["_bullish_trend_raw"] = (
        clean(row.get(bullish_col, None)) if bullish_col else None
    )
    profile["_community_hopes_raw"] = (
        clean(row.get(community_col, None)) if community_col else None
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

    # Find LinkedIn text column — new format uses "Linkedin Profile" for the text dump
    linkedin_col = find_column(df, "Linkedin Profile")
    if linkedin_col is None:
        linkedin_col = find_column(df, "LinkedIn Profile paste")
    if linkedin_col is None:
        print("WARNING: Cannot find LinkedIn profile text column")
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
